#ifndef __COMPAT_H__
#define __COMPAT_H__

#ifdef WIN32

#include <windows.h>

extern int opt_priority;

static __inline void sleep(int secs)
{
	Sleep(secs * 1000);
}

enum {
	PRIO_PROCESS = 0,
};

static __inline int setpriority(int which, int who, int prio)
{
	switch (opt_priority) {
		case 5:
			prio = THREAD_PRIORITY_TIME_CRITICAL;
			break;
		case 4:
			prio = THREAD_PRIORITY_HIGHEST;
			break;
		case 3:
			prio = THREAD_PRIORITY_ABOVE_NORMAL;
			break;
		case 2:
			prio = THREAD_PRIORITY_NORMAL;
			break;
		case 1:
			prio = THREAD_PRIORITY_BELOW_NORMAL;
			break;
		case 0:
		default:
			prio = THREAD_PRIORITY_IDLE;
	}
	return -!SetThreadPriority(GetCurrentThread(), prio);
}

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#endif /* WIN32 */

#endif /* __COMPAT_H__ */
