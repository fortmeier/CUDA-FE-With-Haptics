/** \file haptic.h
 *
 *  \b Initial \b Author: fortmeier
 *  \b Copyright (C) 2012 Institute of Medical Informatics,
 *     University of Luebeck
 *
 * Very simple haptic interaction for prototyping purposes.
 * This file uses the OpenHaptics Toolkit by Sensable:
 *
 *   http://www.sensable.com/products-openhaptics-toolkit.htm
 *
 ****************************************************************************/


#include <stdio.h>
#include <HD/hd.h>
#include <HDU/hduVector.h>
#include <HDU/hduError.h>

float hX;
float hY;
float hZ;

float fX;
float fY;
float fZ;

using namespace std;


HDCallbackCode HDCALLBACK HapticCallback(void *data)
{
    hdBeginFrame(hdGetCurrentDevice());
    hduVector3Dd position, force;
    hdGetDoublev(HD_CURRENT_POSITION, position);

    hX = position[0]/5.;
    hY = position[1]/5.;
    hZ = position[2]/5.;

    HDint nCurrentButtons;

    hdGetIntegerv(HD_CURRENT_BUTTONS, &nCurrentButtons);

    if ((nCurrentButtons & HD_DEVICE_BUTTON_1) != 0)
    {
      force[0] = fX;
      force[1] = fY;
      force[2] = fZ;
    }
    else
    {
      force[0] = 0;
      force[1] = 0;
      force[2] = 0;
    }

    hdSetDoublev(HD_CURRENT_FORCE, force);
    hdEndFrame(hdGetCurrentDevice());
    HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Scheduler callback produced an error \n");
        if (hduIsSchedulerError(&error))
        {
            return HD_CALLBACK_DONE;  
        }
    }
    return HD_CALLBACK_CONTINUE;
}


HHD hHD;
HDCallbackCode hCallback;

int initHaptic()
{
    HDErrorInfo error;

    hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Init of haptic device failed!");
        return -1;
    }
    hdEnable(HD_FORCE_OUTPUT);

    hdSetSchedulerRate(1000);

    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Haptic Scheduler failed to start!");
        return -1;
    }

    hCallback = hdScheduleAsynchronous( HapticCallback, 0, HD_DEFAULT_SCHEDULER_PRIORITY );

    return 0;
}

int deinitHaptic() {
    hdStopScheduler();
    hdUnschedule(hCallback);
    hdDisableDevice(hHD);
    return 0;
}
