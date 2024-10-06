#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"    
#include "bsp_control.h"
#include "bsp_usart.h"
#include <stdio.h>

osThreadId defaultTaskHandle;

void StartDefaultTask(void const * argument);
void MX_FREERTOS_Init(void); 

void MX_FREERTOS_Init(void)
	{
  
	osThreadDef(MainTask, Main_Task, osPriorityNormal, 0, 128);//���˻�����ϵͳ
  osThreadCreate(osThread(MainTask), NULL);
	
	osThreadDef(UARTTask, UART_task, osPriorityNormal, 0, 128);   //ͨѶϵͳ
  osThreadCreate(osThread(UARTTask), NULL);

	}


