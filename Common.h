#pragma once

#if defined(_MSC_VER)

#	if defined(IMFUSION_HISTOCTREG_DLL)
#		define IMFUSION_HISTOCTREG_API __declspec(dllexport)
#	elif defined(IMFUSIONLIB_STATIC)
#		define IMFUSION_HISTOCTREG_API
#	else
#		define IMFUSION_HISTOCTREG_API __declspec(dllimport)
#	endif

#else

#	if defined(IMFUSION_HISTOCTREG_DLL)
#		define IMFUSION_HISTOCTREG_API __attribute__((visibility("default")))
#	elif defined(IMFUSIONLIB_STATIC)
#		define IMFUSION_HISTOCTREG_API
#	else
#		define IMFUSION_HISTOCTREG_API __attribute__((visibility("default")))
#	endif

#endif
