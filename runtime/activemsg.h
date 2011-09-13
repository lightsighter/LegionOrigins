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
