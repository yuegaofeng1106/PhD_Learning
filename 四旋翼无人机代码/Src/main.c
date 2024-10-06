/**
  ******************************************************************************
  * File Name          : main.c
  * Description        : Main program body
  ******************************************************************************
  *
  * COPYRIGHT(c) 2019 STMicroelectronics and ShenYang university laboratory of 327
  *
  * Author:YUE Gaofeng 
  *breif :Written mainly for security patrol vehicle
  ******************************************************************************
  */
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"
#include "cmsis_os.h"
#include "can.h"
#include "usart.h"
#include "gpio.h"
#include "tim.h"
#include "pid.h"
#include "dma.h"
#include "bsp_usart.h"
#include "bsp_can.h"
#include "bsp_control.h"

void SystemClock_Config(void);
void Error_Handler(void);
void MX_FREERTOS_Init(void);

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_CAN1_Init();
	MX_DMA_Init();
	MX_USART1_UART_Init();//DR16接收机通道，接收遥控器指令
	MX_USART2_UART_Init();//备用
  MX_USART3_UART_Init();//
  MX_USART6_UART_Init();//上位机通讯,例如树莓派、妙算等
	MX_USART7_UART_Init();//备用
	MX_USART8_UART_Init();//备用

  MX_TIM1_Init();//投掷器使用
	MX_TIM2_Init();//备用
	MX_TIM4_Init();//四个旋翼电机使用
	MX_TIM12_Init();//蜂鸣器
	Led_init(30);//LED快闪2s左右，证明时钟初始化正确
	Buzzer_init(400);//蜂鸣器打开1s，证明初始化完成
	HAL_Delay(100);	//add , wait device stable, very very important!!!
  MX_FREERTOS_Init();
  osKernelStart();

  while (1)
  {

  }
}

void SystemClock_Config(void)
{

  RCC_OscInitTypeDef RCC_OscInitStruct;
  RCC_ClkInitTypeDef RCC_ClkInitStruct;

  __HAL_RCC_PWR_CLK_ENABLE();

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 6;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }

  HAL_SYSTICK_Config(HAL_RCC_GetHCLKFreq()/1000);

  HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK);

  /* SysTick_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(SysTick_IRQn, 15, 0);
}
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM6) 
	{
    HAL_IncTick();
  }
	
	
}
void Error_Handler(void)
{
  while(1) 
  {
		
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t* file, uint32_t line)
{

}
#endif



