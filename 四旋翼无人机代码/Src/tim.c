/*******************************************************************************
  * File Name          : TIM.c
  * Description        : This file provides code for the configuration
  *                      of the TIM instances.
  ******************************************************************************
  * Includes ------------------------------------------------------------------*/
/*�����Ҽ�����һ�¼�����ʱ��������
��ʱ��1 8 Ϊ�߼���ʱ�� �������ɱ�������������
2 3 4 5 9 10 11 12 13 14Ϊͨ�ö�ʱ������Ҫ����ʱ������ PWM ���벶�� ����Ƚ�
6 7Ϊ������ʱ�� ��������DAC����PWM���
*/
#include "tim.h"
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim12;

/* TIM init function */
/* TIM1 init function */
void MX_TIM1_Init(void)
{ 
  TIM_OC_InitTypeDef sConfigOC;

  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 899;//(899+1)M/90=10M,�Զ���װ��ֵ2500��10M/2500=4kHZ
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 2499;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  HAL_TIM_PWM_Init(&htim1);
 
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;

  HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2);

  HAL_TIM_MspPostInit(&htim1);
	HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
	HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);

	
}
void MX_TIM2_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC;//���������Ժ�Ͳ������ˣ�output compare 
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 899;//Ԥ��Ƶϵ��
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 2499;//�Զ���װ��ֵ
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;//ʱ�ӷ�Ƶ����
  HAL_TIM_PWM_Init(&htim2);//ͨ�ö�ʱ���Ĺ���

  sConfigOC.OCMode = TIM_OCMODE_PWM1;//�Ƚ������ΪPWM1ʹ��
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;

  HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2);
  HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3);
  HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4);
	
  HAL_TIM_MspPostInit(&htim2);//����IO��
		
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);//
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);//
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);//
	HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);//
	
}
 

/* TIM4 init function */
void MX_TIM4_Init(void)
{ 
  TIM_OC_InitTypeDef sConfigOC;

  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 89;//(89+1)M/90=1M,�Զ���װ��ֵ2000��1M/2000=500HZ
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 2000;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  HAL_TIM_PWM_Init(&htim4);
 
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1000;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;

  HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_2);
  HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_3);
  HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_4);
  HAL_TIM_MspPostInit(&htim4);
	HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_1);
	HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_2);
	HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_3);
	HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_4);
}

void MX_TIM12_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC;

  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 89;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 600;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
	HAL_TIM_PWM_Init(&htim12);//ͨ�ö�ʱ���Ĺ���	
	
  sConfigOC.OCMode = TIM_OCMODE_PWM1;	
	
	HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_1);
  HAL_TIM_MspPostInit(&htim12);
	HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_1);
}

void HAL_TIM_PWM_MspInit(TIM_HandleTypeDef* tim_pwmHandle)
{

	if(tim_pwmHandle->Instance==TIM1)
  {
    __HAL_RCC_TIM1_CLK_ENABLE();
  }
	if(tim_pwmHandle->Instance==TIM2)
  {
    __HAL_RCC_TIM2_CLK_ENABLE();
  }

	if(tim_pwmHandle->Instance==TIM4)
  {
    __HAL_RCC_TIM4_CLK_ENABLE();
  }
		if(tim_pwmHandle->Instance==TIM12)
  {
    __HAL_RCC_TIM12_CLK_ENABLE();
  }

}

void HAL_TIM_MspPostInit(TIM_HandleTypeDef* timHandle)
{
GPIO_InitTypeDef GPIO_InitStruct;

	if(timHandle->Instance==TIM1)
  { 
    /**TIM2 GPIO Configuration    
    PA8     ------> TIM1_CH1
		PA9     ------> TIM1_CH2
    */
    GPIO_InitStruct.Pin = GPIO_PIN_8|GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM1;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  }
	if(timHandle->Instance==TIM2)
  { 
    /**TIM2 GPIO Configuration    
    PA0     ------> TIM2_CH1
    PA1     ------> TIM2_CH2
    PA2     ------> TIM2_CH3
    PA3     ------> TIM2_CH4 
    */

    GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  }

	if(timHandle->Instance==TIM4)
  {
	 /**TIM4 GPIO Configuration
		PD15     ------> TIM4_CH4	    
    PD14     ------> TIM4_CH3
    PD13     ------> TIM4_CH2
    PD12     ------> TIM4_CH1 
    */
    GPIO_InitStruct.Pin = GPIO_PIN_15|GPIO_PIN_14|GPIO_PIN_13|GPIO_PIN_12;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF2_TIM4;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
	}
	
 if(timHandle->Instance==TIM12)
  { 
    /**TIM12 GPIO Configuration    
    PH6     ------> TIM12_CH1 
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF9_TIM12;
    HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);
	
  }
  
}

void Buzzer_init(int pre)
{
	__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_1,pre);
	HAL_Delay(170);
	__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_1,pre-400);
	HAL_Delay(170);
	__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_1,pre);
	HAL_Delay(170);
	__HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_1,pre-400);
	HAL_Delay(170);	
	__HAL_TIM_SET_COMPARE(&htim12 ,TIM_CHANNEL_1,400);	
	HAL_Delay(170);	
	__HAL_TIM_SET_COMPARE(&htim12 ,TIM_CHANNEL_1,0);
	HAL_Delay(100);
}


