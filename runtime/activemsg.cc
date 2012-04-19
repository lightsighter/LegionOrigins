#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

#include <queue>
#include <cassert>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

enum { MSGID_FLIP_REQ = 254,
       MSGID_FLIP_ACK = 255 };

struct OutgoingMessage {
  OutgoingMessage(unsigned _msgid, unsigned _num_args, const void *_args)
    : msgid(_msgid), num_args(_num_args),
      payload(0), payload_size(0), payload_mode(PAYLOAD_NONE)
  {
    for(unsigned i = 0; i < _num_args; i++)
      args[i] = ((const int *)_args)[i];
  }

  ~OutgoingMessage(void)
  {
    if((payload_mode == PAYLOAD_COPY) || (payload_mode == PAYLOAD_FREE)) {
      assert(payload_size > 0);
      free(payload);
    }
  }

  void set_payload(void *_payload, size_t _payload_size, int _payload_mode)
  {
    if(_payload_mode != PAYLOAD_NONE) {
      payload_mode = _payload_mode;
      payload_size = _payload_size;
      if(payload_mode == PAYLOAD_COPY) {
	payload = malloc(payload_size);
	memcpy(payload, _payload, payload_size);
      } else 
	payload = _payload;
    }
  }

  unsigned msgid;
  unsigned num_args;
  void *payload;
  size_t payload_size;
  int payload_mode;
  int args[16];
};
    
class ActiveMessageEndpoint {
public:
  static const int NUM_LMBS = 2;
  static const size_t LMB_SIZE = (4 << 20);

  ActiveMessageEndpoint(gasnet_node_t _peer, const gasnet_seginfo_t *seginfos)
    : peer(_peer)
  {
    gasnet_hsl_init(&mutex);

    cur_write_lmb = 0;
    cur_write_offset = 0;
    cur_write_count = 0;

    for(int i = 0; i < NUM_LMBS; i++) {
      lmb_w_bases[i] = ((char *)(seginfos[peer].addr)) + (seginfos[peer].size - LMB_SIZE * (gasnet_mynode() * NUM_LMBS + i + 1));
      lmb_r_bases[i] = ((char *)(seginfos[gasnet_mynode()].addr)) + (seginfos[peer].size - LMB_SIZE * (peer * NUM_LMBS + i + 1));
      lmb_r_counts[i] = 0;
      lmb_w_avail[i] = true;
    }
  }

  int push_messages(int max_to_send = 0)
  {
    int count = 0;

    while((max_to_send == 0) || (count < max_to_send)) {
      // attempt to get the mutex that covers the outbound queues - do not
      //  block
      int ret = gasnet_hsl_trylock(&mutex);
      if(ret == GASNET_ERR_NOT_READY) break;

      // try to send a long message, but only if we have an LMB available
      //  on the receiving end
      if((out_long_hdrs.size() > 0) && lmb_w_avail[cur_write_lmb]) {
	OutgoingMessage *hdr;
	hdr = out_long_hdrs.front();

	// do we have enough room in the current LMB?
	assert(hdr->payload_size <= LMB_SIZE);
	if((cur_write_offset + hdr->payload_size) <= LMB_SIZE) {
	  // we can send the message - update lmb pointers and remove the
	  //  packet from the queue, and then drop them mutex before
	  //  sending the message
	  char *dest_ptr = lmb_w_bases[cur_write_lmb] + cur_write_offset;
	  cur_write_offset += hdr->payload_size;
	  cur_write_count++;
	  out_long_hdrs.pop();

	  gasnet_hsl_unlock(&mutex);

	  send_long(hdr, dest_ptr);
	  delete hdr;
	  count++;
	  continue;
	} else {
	  // can't send the message, so flip the buffer that's now full
	  int flip_buffer = cur_write_lmb;
	  int flip_count = cur_write_count;
	  lmb_w_avail[cur_write_lmb] = false;
	  cur_write_lmb = (cur_write_lmb + 1) % NUM_LMBS;
	  cur_write_offset = 0;
	  cur_write_count = 0;

	  // now let go of the lock and send the flip request
	  gasnet_hsl_unlock(&mutex);

	  gasnet_AMRequestShort2(peer, MSGID_FLIP_REQ,
				 flip_buffer, flip_count);

	  continue;
	}
      }

      // couldn't send a long message, try a short message
      if(out_short_hdrs.size() > 0) {
	OutgoingMessage *hdr = out_short_hdrs.front();
	out_short_hdrs.pop();

	// now let go of lock and send message
	gasnet_hsl_unlock(&mutex);

	send_short(hdr);
	delete hdr;
	count++;
	continue;
      }

      // if we get here, we didn't find anything to do, so break out of loop
      //  after releasing the lock
      gasnet_hsl_unlock(&mutex);
      break;
    }

    return count;
  }

  void enqueue_message(OutgoingMessage *hdr)
  {
    // need to hold the mutex in order to push onto one of the queues
    gasnet_hsl_lock(&mutex);

    if(hdr->payload_size > gasnet_AMMaxMedium())
      out_long_hdrs.push(hdr);
    else
      out_short_hdrs.push(hdr);

    gasnet_hsl_unlock(&mutex);
  }

  void handle_long_msgptr(void *ptr)
  {
    // can figure out which buffer it is without holding lock
    int r_buffer = -1;
    for(int i = 0; i < NUM_LMBS; i++)
      if((ptr >= lmb_r_bases[i]) && (ptr < (lmb_r_bases[i] + LMB_SIZE))) {
      r_buffer = i;
      break;
    }
    assert(r_buffer >= 0);

    // now take the lock to increment the r_count and decide if we need
    //  to ack (can't actually send it here, so queue it up)
    gasnet_hsl_lock(&mutex);
    lmb_r_counts[r_buffer]++;
    if(lmb_r_counts[r_buffer] == 0) {
      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &r_buffer);
      out_short_hdrs.push(hdr);
    }
    gasnet_hsl_unlock(&mutex);
  }

  // called when the remote side tells us that there will be no more
  //  messages sent for a given buffer - as soon as we've received them all,
  //  we can ack
  void handle_flip_request(int buffer, int count)
  {
    gasnet_hsl_lock(&mutex);
    lmb_r_counts[buffer] -= count;
    if(lmb_r_counts[buffer] == 0) {
      OutgoingMessage *hdr = new OutgoingMessage(MSGID_FLIP_ACK, 1, &buffer);
      out_short_hdrs.push(hdr);
    }
    gasnet_hsl_unlock(&mutex);
  }

  // called when the remote side says it has received all the messages in a
  //  given buffer - we can that mark that write buffer as available again
  //  (don't even need to take the mutex!)
  void handle_flip_ack(int buffer)
  {
    lmb_w_avail[buffer] = true;
  }

protected:
  void send_short(OutgoingMessage *hdr)
  {
    switch(hdr->num_args) {
    case 1:
      if(hdr->payload_size)
	gasnet_AMRequestMedium1(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0]);
      else
	gasnet_AMRequestShort1(peer, hdr->msgid, hdr->args[0]);
      break;

    case 2:
      if(hdr->payload_size)
	gasnet_AMRequestMedium2(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1]);
      else
	gasnet_AMRequestShort2(peer, hdr->msgid, hdr->args[0], hdr->args[1]);
      break;

    case 3:
      if(hdr->payload_size)
	gasnet_AMRequestMedium3(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2]);
      else
	gasnet_AMRequestShort3(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2]);
      break;

    case 4:
      if(hdr->payload_size)
	gasnet_AMRequestMedium4(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3]);
      else
	gasnet_AMRequestShort4(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3]);
      break;

    case 5:
      if(hdr->payload_size)
	gasnet_AMRequestMedium5(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4]);
      else
	gasnet_AMRequestShort5(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4]);
      break;

    case 6:
      if(hdr->payload_size)
	gasnet_AMRequestMedium6(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5]);
      else
	gasnet_AMRequestShort6(peer, hdr->msgid,
			       hdr->args[0], hdr->args[1], hdr->args[2],
			       hdr->args[3], hdr->args[4], hdr->args[5]);
      break;

    case 12:
      if(hdr->payload_size)
	gasnet_AMRequestMedium12(peer, hdr->msgid, hdr->payload, hdr->payload_size,
				 hdr->args[0], hdr->args[1], hdr->args[2],
				 hdr->args[3], hdr->args[4], hdr->args[5],
				 hdr->args[6], hdr->args[7], hdr->args[8],
				 hdr->args[9], hdr->args[10], hdr->args[11]);
      else
	gasnet_AMRequestShort12(peer, hdr->msgid,
				hdr->args[0], hdr->args[1], hdr->args[2],
				hdr->args[3], hdr->args[4], hdr->args[5],
				hdr->args[6], hdr->args[7], hdr->args[8],
				hdr->args[9], hdr->args[10], hdr->args[11]);
      break;

    default:
      fprintf(stderr, "need to support short/medium of size=%d\n", hdr->num_args);
      assert(1==2);
    }
  }
  
  void send_long(OutgoingMessage *hdr, void *dest_ptr)
  {
    switch(hdr->num_args) {
    case 1:
      gasnet_AMRequestLong1(peer, hdr->msgid, 
			    hdr->payload, hdr->payload_size, dest_ptr,
			    hdr->args[0]);
      break;

    case 2:
      gasnet_AMRequestLong2(peer, hdr->msgid, 
			    hdr->payload, hdr->payload_size, dest_ptr,
			    hdr->args[0], hdr->args[1]);
      break;

    case 3:
      gasnet_AMRequestLong3(peer, hdr->msgid, 
			    hdr->payload, hdr->payload_size, dest_ptr,
			    hdr->args[0], hdr->args[1], hdr->args[2]);
      break;

    case 6:
      gasnet_AMRequestLong6(peer, hdr->msgid, 
			    hdr->payload, hdr->payload_size, dest_ptr,
			    hdr->args[0], hdr->args[1], hdr->args[2],
			    hdr->args[3], hdr->args[4], hdr->args[5]);
      break;

    default:
      fprintf(stderr, "need to support long of size=%d\n", hdr->num_args);
      assert(3==4);
    }
  }

  gasnet_node_t peer;
  
  gasnet_hsl_t mutex;
  std::queue<OutgoingMessage *> out_short_hdrs;
  std::queue<OutgoingMessage *> out_long_hdrs;

  int cur_write_lmb, cur_write_count;
  size_t cur_write_offset;
  char *lmb_w_bases[NUM_LMBS];
  char *lmb_r_bases[NUM_LMBS];
  int lmb_r_counts[NUM_LMBS];
  bool lmb_w_avail[NUM_LMBS];
};

static ActiveMessageEndpoint **endpoints;

static void handle_flip_req(gasnet_token_t token,
		     int flip_buffer, int flip_count)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);
  endpoints[src]->handle_flip_request(flip_buffer, flip_count);
}

static void handle_flip_ack(gasnet_token_t token,
			    int ack_buffer)
{
  gasnet_node_t src;
  gasnet_AMGetMsgSource(token, &src);
  endpoints[src]->handle_flip_ack(ack_buffer);
}

void init_endpoints(gasnet_handlerentry_t *handlers, int hcount,
		    int gasnet_mem_size_in_mb)
{
  // add in our internal handlers and space we need for LMBs
  int attach_size = ((gasnet_mem_size_in_mb << 20) +
		     (gasnet_nodes() * 
		      ActiveMessageEndpoint::NUM_LMBS *
		      ActiveMessageEndpoint::LMB_SIZE));

  handlers[hcount].index = MSGID_FLIP_REQ;
  handlers[hcount].fnptr = (void (*)())handle_flip_req;
  hcount++;
  handlers[hcount].index = MSGID_FLIP_ACK;
  handlers[hcount].fnptr = (void (*)())handle_flip_ack;
  hcount++;

  CHECK_GASNET( gasnet_attach(handlers, hcount,
			      attach_size, 0) );

  endpoints = new ActiveMessageEndpoint *[gasnet_nodes()];

  gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
  CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );

  for(int i = 0; i < gasnet_nodes(); i++)
    if(i == gasnet_mynode())
      endpoints[i] = 0;
    else
      endpoints[i] = new ActiveMessageEndpoint(i, seginfos);

  delete[] seginfos;
}

static int num_polling_threads = 0;
static pthread_t *polling_threads = 0;

// do a little bit of polling to try to move messages along, but return
//  to the caller rather than spinning
void do_some_polling(void)
{
  for(int i = 0; i < gasnet_nodes(); i++) {
    if(!endpoints[i]) continue; // skip our own node

    endpoints[i]->push_messages(0);
  }

  gasnet_AMPoll();
}

static void *gasnet_poll_thread_loop(void *data)
{
  // each polling thread basically does an endless loop of trying to send
  //  outgoing messages and then polling
  while(1) {
    do_some_polling();
    //usleep(10000);
  }
  return 0;
}

void start_polling_threads(int count)
{
  num_polling_threads = count;
  polling_threads = new pthread_t[count];

  for(int i = 0; i < count; i++)
    CHECK_PTHREAD( pthread_create(&polling_threads[i], 0, 
				  gasnet_poll_thread_loop, 0) );
}
	
void enqueue_message(gasnet_node_t target, int msgid,
		     const void *args, size_t arg_size,
		     const void *payload, size_t payload_size,
		     int payload_mode)
{
  assert(target != gasnet_mynode());

  OutgoingMessage *hdr = new OutgoingMessage(msgid, 
					     (arg_size + sizeof(int) - 1) / sizeof(int),
					     args);

  hdr->set_payload((void *)payload, payload_size, payload_mode);

  endpoints[target]->enqueue_message(hdr);
}

void handle_long_msgptr(gasnet_node_t source, void *ptr)
{
  assert(source != gasnet_mynode());

  endpoints[source]->handle_long_msgptr(ptr);
}
