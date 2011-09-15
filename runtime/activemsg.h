// hopefully a more user-friendly C++ template wrapper for GASNet active
//  messages...

#ifndef ACTIVEMSG_H
#define ACTIVEMSG_H

template <class T, void (*FNPTR)(T), int N> struct HandlerRawArgs;

template <class T, void (*FNPTR)(T), int N> struct HandlerArgUnion {
public:
  union {
    HandlerRawArgs<T,FNPTR,N> raw;
    T                         typed;
  } args;
};

template <class T> struct HandlerReplyFuture {
  gasnet_hsl_t mutex;
  gasnett_cond_t condvar;
  bool valid;
  T value;

  HandlerReplyFuture(void) {
    gasnet_hsl_init(&mutex);
    gasnett_cond_init(&condvar);
    valid = false;
  }

  void set(T newval)
  {
    gasnet_hsl_lock(&mutex);
    valid = true;
    value = newval;
    gasnett_cond_broadcast(&condvar);
    gasnet_hsl_unlock(&mutex);
  }

  bool is_set(void) const { return valid; }

  void wait(void)
  {
    gasnet_hsl_lock(&mutex);
    while(!valid) gasnett_cond_wait(&condvar, &mutex.lock);
    gasnet_hsl_unlock(&mutex);
  }

  T get(void) const { return value; }
};

template <class ARGTYPE, class RPLTYPE>
struct ArgsWithReplyInfo {
  HandlerReplyFuture<RPLTYPE> *fptr;
  ARGTYPE                      args;

  /* ArgsWithReplyInfo(HandlerReplyFuture<RPLTYPE> *_fptr, ARGTYPE _args) */
  /* : fptr(_fptr), args(_args) {} */
};


/* template <class T, void (*FNPTR)(T), int N> struct HandlerArgReplyUnion { */
/* public: */
/*   union { */
/*     HandlerRawArgs<T,FNPTR,N> raw; */
/*     struct { */
/*       HandlerReplyFuture<T>  *future; */
/*       T                       typed; */
/*     }; */
/*   } args; */
/* }; */

template <class RPLTYPE, int RPLID, int RPL_N> struct ReplyRawArgs;

template <class RPLTYPE, int RPLID> struct ReplyRawArgs<RPLTYPE, RPLID, 1> {
  gasnet_handlerarg_t arg0;

  void reply_short(gasnet_token_t token)
  {
    gasnet_AMReplyShort1(token, RPLID, arg0);
  }

  static void handler_short(gasnet_token_t token, 
			    gasnet_handlerarg_t arg0)
  {
    gasnet_node_t src;
    gasnet_AMGetMsgSource(token, &src);
    printf("%d: handling reply from node %d\n", (int)gasnet_mynode(), src);
    union {
      ReplyRawArgs<RPLTYPE,RPLID,1> raw;
      /* RequestRawArgs<REQTYPE, REQID, RPLTYE, RPLID, FNPTR, */
      /* 	             sizeof(RPLTYPE)/4, sizeof(REQTYPE)/4> raw; */
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed;
    } u;
    u.args.raw.arg0 = arg0;
    u.args.typed.fptr->set(u.args.typed.args);
  }
};

#define HANDLERARG_DECL_1                     gasnet_handlerarg_t arg0
#define HANDLERARG_DECL_2  HANDLERARG_DECL_1; gasnet_handlerarg_t arg1
#define HANDLERARG_DECL_3  HANDLERARG_DECL_2; gasnet_handlerarg_t arg2

#define HANDLERARG_VALS_1                     arg0
#define HANDLERARG_VALS_2  HANDLERARG_VALS_1, arg1
#define HANDLERARG_VALS_3  HANDLERARG_VALS_2, arg2

#define HANDLERARG_PARAMS_1                      gasnet_handlerarg_t arg0
#define HANDLERARG_PARAMS_2 HANDLERARG_PARAMS_1, gasnet_handlerarg_t arg1
#define HANDLERARG_PARAMS_3 HANDLERARG_PARAMS_2, gasnet_handlerarg_t arg2

#define HANDLERARG_COPY_1                    u.raw.arg0 = arg0
#define HANDLERARG_COPY_2 HANDLERARG_COPY_1; u.raw.arg1 = arg1
#define HANDLERARG_COPY_3 HANDLERARG_COPY_2; u.raw.arg2 = arg2

#define MACROPROXY(a,...) a(__VA_ARGS__)

#define REPLY_RAW_ARGS(n) \
template <class RPLTYPE, int RPLID> struct ReplyRawArgs<RPLTYPE, RPLID, n> { \
  HANDLERARG_DECL_ ## n ; \
\
  void reply_short(gasnet_token_t token) \
  { \
    MACROPROXY(gasnet_AMReplyShort ## n, token, RPLID, HANDLERARG_VALS_ ## n ); \
  } \
 \
  static void handler_short(gasnet_token_t token, HANDLERARG_PARAMS_ ## n ) \
  { \
    gasnet_node_t src; \
    gasnet_AMGetMsgSource(token, &src); \
    printf("%d: handling reply from node %d\n", (int)gasnet_mynode(), src); \
    union { \
      ReplyRawArgs<RPLTYPE,RPLID,n> raw; \
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed; \
    } u; \
    HANDLERARG_COPY_ ## n ; \
    u.typed.fptr->set(u.typed.args); \
  } \
}

REPLY_RAW_ARGS(3);

#if 0
template <class RPLTYPE, int RPLID> struct ReplyRawArgs<RPLTYPE, RPLID, 3> {
  gasnet_handlerarg_t arg0;
  gasnet_handlerarg_t arg1;
  gasnet_handlerarg_t arg2;

  void reply_short(gasnet_token_t token)
  {
    gasnet_AMReplyShort3(token, RPLID, arg0, arg1, arg2);
  }

  static void handler_short(gasnet_token_t token, 
			    gasnet_handlerarg_t arg0, 
			    gasnet_handlerarg_t arg1, 
			    gasnet_handlerarg_t arg2)
  {
    gasnet_node_t src;
    gasnet_AMGetMsgSource(token, &src);
    printf("%d: handling reply from node %d\n", (int)gasnet_mynode(), src);
    union {
      ReplyRawArgs<RPLTYPE,RPLID,3> raw;
      /* RequestRawArgs<REQTYPE, REQID, RPLTYE, RPLID, FNPTR, */
      /* 	             sizeof(RPLTYPE)/4, sizeof(REQTYPE)/4> raw; */
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed;
    } u;
    u.raw.arg0 = arg0;
    u.raw.arg1 = arg1;
    u.raw.arg2 = arg2;
    u.typed.fptr->set(u.typed.args);
  }
};
#endif

template <class REQTYPE, int REQID, class RPLTYPE, int RPLID,
          RPLTYPE (*FNPTR)(REQTYPE), int RPL_N, int REQ_N> struct RequestRawArgs;

template <class REQTYPE, int REQID, class RPLTYPE, int RPLID,
          RPLTYPE (*FNPTR)(REQTYPE), int RPL_N>
struct RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, FNPTR, RPL_N, 1> {
  gasnet_handlerarg_t arg0;

  void request_short(gasnet_node_t dest)
  {
    gasnet_AMRequestShort1(dest, REQID, arg0);
  }

  static void handler_short(gasnet_token_t token,
			    gasnet_handlerarg_t arg0)
  {
    gasnet_node_t src;
    gasnet_AMGetMsgSource(token, &src);
    printf("handling request from node %d\n", src);
    union {
      RequestRawArgs<REQTYPE,REQID,RPLTYPE,RPLID,FNPTR,RPL_N,1> raw;
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
    u.args.raw.arg0 = arg0;

    union {
      ReplyRawArgs<RPLTYPE,RPLID,RPL_N> raw;
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed;
    } rpl_u;

    rpl_u.args.typed.args = (*FNPTR)(u.args.typed.args);
    rpl_u.args.typed.fptr = u.args.typed.fptr;
    rpl_u.raw.reply_short(token);
  }
};

template <class REQTYPE, int REQID, class RPLTYPE, int RPLID,
          RPLTYPE (*FNPTR)(REQTYPE), int RPL_N>
struct RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, FNPTR, RPL_N, 3> {
  gasnet_handlerarg_t arg0;
  gasnet_handlerarg_t arg1;
  gasnet_handlerarg_t arg2;

  void request_short(gasnet_node_t dest)
  {
    gasnet_AMRequestShort3(dest, REQID, arg0, arg1, arg2);
  }

  static void handler_short(gasnet_token_t token,
			    gasnet_handlerarg_t arg0,
			    gasnet_handlerarg_t arg1,
			    gasnet_handlerarg_t arg2)
  {
    gasnet_node_t src;
    gasnet_AMGetMsgSource(token, &src);
    printf("handling request from node %d\n", src);
    union {
      RequestRawArgs<REQTYPE,REQID,RPLTYPE,RPLID,FNPTR,RPL_N,3> raw;
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
    u.raw.arg0 = arg0;
    u.raw.arg1 = arg1;
    u.raw.arg2 = arg2;

    union {
      ReplyRawArgs<RPLTYPE,RPLID,RPL_N> raw;
      ArgsWithReplyInfo<RPLTYPE,RPLTYPE> typed;
    } rpl_u;

    rpl_u.typed.args = (*FNPTR)(u.typed.args);
    rpl_u.typed.fptr = u.typed.fptr;
    rpl_u.raw.reply_short(token);
  }
};

template <class T, void (*FNPTR)(T)> struct HandlerRawArgs<T, FNPTR, 1> {
  gasnet_handlerarg_t arg0;

  void request_short(gasnet_node_t dest, gasnet_handler_t handler)
  {
    gasnet_AMRequestShort1(dest, handler, arg0);
  }

  void request_medium(gasnet_node_t dest, gasnet_handler_t handler,
		      const void *data, size_t datalen)
  {
    gasnet_AMRequestMedium1(dest, handler, (void *)data, datalen, arg0);
  }

  /* void request_long(gasnet_node_t dest, gasnet_handler_t handler, */
  /* 		    const void *data, size_t datalen) */
  /* { */
  /*   gasnet_AMRequestLong1(dest, handler, data, datalen, arg0); */
  /* } */

  static void handler_short(gasnet_token_t token,
			    gasnet_handlerarg_t arg0)
  {
    gasnet_node_t src;
    gasnet_AMGetMsgSource(token, &src);
    printf("handling message from node %d\n", src);
    HandlerArgUnion<T,FNPTR,1> u;
    u.args.raw.arg0 = arg0;
    (*FNPTR)(u.args.typed);
  }
};

template <int MSGID, class ARGTYPE, void (*FNPTR)(ARGTYPE)>
class ActiveMessageShortNoReply {
 public:
  typedef HandlerRawArgs<ARGTYPE, FNPTR, sizeof(ARGTYPE)/4> RawArgsType;
  typedef HandlerArgUnion<ARGTYPE, FNPTR, sizeof(ARGTYPE)/4> ArgUnionType;

  static void request(gasnet_node_t dest, ARGTYPE args)
  {
    ArgUnionType u;
    u.args.typed = args;
    u.args.raw.request_short(dest, MSGID);
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    entries[0].index = MSGID;
    entries[0].fnptr = (void (*)()) (RawArgsType::handler_short);
    return 1;
  }
};

template <int REQID, int RPLID, class REQTYPE, class RPLTYPE,
          RPLTYPE (*FNPTR)(REQTYPE)>
class ActiveMessageShortReply {
 public:
  /* typedef ArgsWithReplyInfo<REQTYPE,RPLTYPE> RequestArgsWithInfo; */
  /* typedef ArgsWithReplyInfo<RPLTYPE,RPLTYPE> ReplyArgsWithInfo; */

  typedef RequestRawArgs<REQTYPE, REQID, RPLTYPE, RPLID, FNPTR,
                         (sizeof(void*)+sizeof(RPLTYPE)+3)/4,
                         (sizeof(void*)+sizeof(REQTYPE)+3)/4> ReqRawArgsType;
  typedef ReplyRawArgs<RPLTYPE, RPLID, (sizeof(void*)+sizeof(RPLTYPE)+3)/4> RplRawArgsType;

  static RPLTYPE request(gasnet_node_t dest, REQTYPE args)
  {
    HandlerReplyFuture<RPLTYPE> future;
    union {
      ReqRawArgsType raw;
      /* RequestRawArgs<REQTYPE, REQID, RPLTYE, RPLID, FNPTR, */
      /* 	             sizeof(RPLTYPE)/4, sizeof(REQTYPE)/4> raw; */
      ArgsWithReplyInfo<REQTYPE,RPLTYPE> typed;
    } u;
      
    u.typed.fptr = &future;
    u.typed.args = args;
    u.raw.request_short(dest);

    printf("request sent - waiting for response\n");
    future.wait();
    return future.value;
  }

  static int add_handler_entries(gasnet_handlerentry_t *entries)
  {
    entries[0].index = REQID;
    entries[0].fnptr = (void (*)()) (ReqRawArgsType::handler_short);
    entries[1].index = RPLID;
    entries[1].fnptr = (void (*)()) (RplRawArgsType::handler_short);
    return 2;
  }
};

/* template <int MSGID, class ARGTYPE, void (*FNPTR)(ARGTYPE, const void *, size_t)> */
/* class ActiveMessageMedLongNoReply { */
/*  public: */
/*   static void request(gasnet_node_t dest, gasnet_handler_t handler, */
/* 		      ARGTYPE args, const void *data, size_t datalen) */
/*   { */
/*     HandlerArgUnion<ARGTYPE, sizeof(ARGTYPE)/4> u; */
/*     u.typed = args; */
/*     u.raw.request_medium(dest, handler, data, datalen); */
/*   } */

/*   static int add_handler_entries(gasnet_handlerentry_t *entries) */
/*   { */
/*     entries[0].index = MSGID; */
/*     entries[0].fnptr = FNPTR; */
/*     return 1; */
/*   } */
/* }; */

#endif
