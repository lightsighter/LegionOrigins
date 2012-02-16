
#ifndef __RUNTIME_UTILITIES_H__
#define __RUNTIME_UTILITIES_H__

#include <iostream>
#include <string>

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdarg>

#include <map>
#include <vector>

// outside of namespace because 50-letter-long enums are annoying
enum {
  TIME_NONE,
  TIME_KERNEL,
  TIME_COPY,
  TIME_HIGH_LEVEL,
  TIME_LOW_LEVEL,
  TIME_MAPPER,
  TIME_SYSTEM,
};

#define DETAILED_TIMING

namespace RegionRuntime {
  // widget for generating debug/info messages
  enum LogLevel {
    LEVEL_SPEW,
    LEVEL_DEBUG,
    LEVEL_INFO,
    LEVEL_WARNING,
    LEVEL_ERROR,
    LEVEL_NONE,
  };
#ifndef COMPILE_TIME_MIN_LEVEL
#define COMPILE_TIME_MIN_LEVEL LEVEL_SPEW
#endif

  /**
   * A logger class for tracking everything from debug messages
   * to error messages.
   */
  class Logger {
  public:
    static void init(int argc, const char *argv[])
    {
      for(std::vector<bool>::iterator it = Logger::get_log_cats_enabled().begin();
          it != Logger::get_log_cats_enabled().end();
          it++)
        (*it) = true;

      for(int i = 1; i < argc; i++) {
        if(!strcmp(argv[i], "-level")) {
          Logger::get_log_level() = (LogLevel)atoi(argv[++i]);
          continue;
        }

        if(!strcmp(argv[i], "-cat")) {
          const char *p = argv[++i];

          if(*p == '*') {
            p++;
          } else {
            // turn off all the bits and then we'll turn on only what's requested
            for(std::vector<bool>::iterator it = Logger::get_log_cats_enabled().begin();
                it != Logger::get_log_cats_enabled().end();
                it++)
              (*it) = false;
          }

          while(*p == ',') p++;
          while(*p) {
            bool enable = true;
            if(*p == '-') {
              enable = false;
              p++;
            }
            const char *p2 = p; while(*p2 && (*p2 != ',')) p2++;
            std::string name(p, p2);
            std::map<std::string, int>::iterator it = Logger::get_categories_by_name().find(name);
            if(it == Logger::get_categories_by_name().end()) {
              fprintf(stderr, "unknown log category '%s'!\n", name.c_str());
              exit(1);
            }

            Logger::get_log_cats_enabled()[it->second] = enable;

            p = p2;
            while(*p == ',') p++;
          }
        }
        continue;
      }
#if 0
      printf("logger settings: level=%d cats=", log_level);
      bool first = true;
      for(unsigned i = 0; i < log_cats_enabled.size(); i++)
        if(log_cats_enabled[i]) {
          if(!first) printf(",");
          first = false;
          printf("%s", categories_by_id[i].c_str());
        }
      printf("\n");
#endif
    }

    static inline void log(LogLevel level, int category, const char *fmt, ...)
    {
      if(level >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
        if(level >= Logger::get_log_level()) {             // dynamic opt-out
          if(Logger::get_log_cats_enabled()[category]) {   // category filter
            va_list args;
            va_start(args, fmt);
            logvprintf(level, category, fmt, args);
            va_end(args);
          }
        }
      }
    }

    class Category {
    public:
      Category(const std::string& name)
      {
        index = Logger::add_category(name);
      }

      operator int(void) const { return index; }

      int index;

      inline void operator()(int level, const char *fmt, ...) __attribute__((format (printf, 3, 4)))
      {
        if(level >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(level >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf((LogLevel)level, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void spew(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_SPEW >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_SPEW >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_SPEW, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void debug(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_DEBUG >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_DEBUG >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_DEBUG, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void info(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_INFO >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_INFO >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_INFO, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void warning(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_WARNING >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_WARNING >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_WARNING, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void error(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_ERROR >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_ERROR >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_ERROR, index, fmt, args);
              va_end(args);
            }
          }
        }
      }
    };

  protected:
    // Implementations specific to low level runtime
    static void logvprintf(LogLevel level, int category, const char *fmt, va_list args);
    static int add_category(const std::string& name)
    {
      int index;
      std::map<std::string, int>::iterator it = Logger::get_categories_by_name().find(name);
      if(it == Logger::get_categories_by_name().end()) {
        index = Logger::get_categories_by_id().size();
        Logger::get_categories_by_id().push_back(name);
        Logger::get_categories_by_name()[name] = index;
        Logger::get_log_cats_enabled().resize(index + 1);
      } else {
        index = it->second;
      }
      return index;
    }
    // Use static methods with static variables to avoid initialization ordering problem
    static LogLevel& get_log_level(void)
    {
      // default level for now is INFO
      static LogLevel log_level = LEVEL_INFO;
      return log_level;
    }
    static std::vector<bool>& get_log_cats_enabled(void)
    {
      static std::vector<bool> log_cats_enabled;
      return log_cats_enabled;
    }
    static std::map<std::string,int>& get_categories_by_name(void)
    {
      static std::map<std::string,int> categories_by_name;
      return categories_by_name;
    }
    static std::vector<std::string>& get_categories_by_id(void)
    {
      static std::vector<std::string> categories_by_id;
      return categories_by_id;
    }
    // Conversion from a log level to a string representation
    static const char* stringify(LogLevel level)
    {
      switch (level)
      {
        case LEVEL_SPEW:
          return "SPEW";
        case LEVEL_DEBUG:
          return "DEBUG";
        case LEVEL_INFO:
          return "INFO";
        case LEVEL_WARNING:
          return "WARNING";
        case LEVEL_ERROR:
          return "ERROR";
        case LEVEL_NONE:
          return "NONE";
      }
      assert(false);
      return NULL;
    }
  };

  /**
   * A timer class for doing detailed timing analysis of applications
   * Implementation is specific to low level runtimes
   */
  namespace LowLevel {
    class DetailedTimer {
    public:
#ifdef DETAILED_TIMING
      static void clear_timers(void);
      static void push_timer(int timer_kind);
      static void pop_timer(void);
      static void roll_up_timers(std::map<int, double>& timers, bool local_only);
      static void report_timers(bool local_only = false);
#else
      static void clear_timers(void) {}
      static void push_timer(int timer_kind) {}
      static void pop_timer(void) {}
      static void roll_up_timers(std::map<int, double>& timers, bool local_only) {}
      static void report_timers(bool local_only = false) {}
#endif
      class ScopedPush {
      public:
        ScopedPush(int timer_kind) { push_timer(timer_kind); }
        ~ScopedPush(void) { pop_timer(); }
      };
    };
  };

  // typedef so we can use detailed timers anywhere in the runtime
  typedef LowLevel::DetailedTimer DetailedTimer;
};

#endif // __RUNTIME_UTILITIES_H__
