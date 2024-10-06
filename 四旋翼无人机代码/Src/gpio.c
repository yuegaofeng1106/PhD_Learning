#include "gpio.h"
#define LED_H_Pin GPIO_PIN_8
#define LED_H_GPIO_Port GPIOG
#define LED_G_Pin GPIO_PIN_7
#define LED_G_GPIO_Port GPIOG
#define LED_F_Pin GPIO_PIN_6
#define LED_F_GPIO_Port GPIOG
#define LED_E_Pin GPIO_PIN_5
#define LED_E_GPIO_Port GPIOG
#define LED_D_Pin GPIO_PIN_4
#define LED_D_GPIO_Port GPIOG
#define LED_C_Pin GPIO_PIN_3
#define LED_C_GPIO_Port GPIOG
#define LED_B_Pin GPIO_PIN_2
#define LED_B_GPIO_Port GPIOG
#define LED_A_Pin GPIO_PIN_1
#define LED_A_GPIO_Port GPIOG

void LED_PWM(void)
{
	 HAL_GPIO_TogglePin(LED_A_GPIO_Port, LED_A_Pin);
	 HAL_GPIO_TogglePin(LED_B_GPIO_Port, LED_B_Pin);
	 HAL_GPIO_TogglePin(LED_C_GPIO_Port, LED_C_Pin);
	 HAL_GPIO_TogglePin(LED_D_GPIO_Port, LED_D_Pin);
	 HAL_GPIO_TogglePin(LED_E_GPIO_Port, LED_E_Pin);
	 HAL_GPIO_TogglePin(LED_F_GPIO_Port, LED_F_Pin);
	 HAL_GPIO_TogglePin(LED_G_GPIO_Port, LED_G_Pin);
	 HAL_GPIO_TogglePin(LED_H_GPIO_Port, LED_H_Pin);
	 HAL_Delay(1);
}
void MX_GPIO_Init(void)
{

  GPIO_InitTypeDef GPIO_InitStruct;

  /* GPIO Ports Clock Enable */ 
	__HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOE_CLK_ENABLE();
	__HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();
	

  /*Configure GPIO pin : PG13 ¼¤¹â*/
  GPIO_InitStruct.Pin = LASER_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LASER_GPIO_Port, &GPIO_InitStruct);
  /*Configure GPIO pin : 
	PE11   ÂÌµÆ
	PF14   ºìµÆ
	*/
  GPIO_InitStruct.Pin = RED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(RED_GPIO_Port, &GPIO_InitStruct);
	
  GPIO_InitStruct.Pin = GREEN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GREEN_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LED_H_Pin|LED_G_Pin|LED_F_Pin|LED_E_Pin 
                          |LED_D_Pin|LED_C_Pin|LED_B_Pin|LED_A_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);
	
	/*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, LED_H_Pin|LED_G_Pin|LED_F_Pin|LED_E_Pin 
                          |LED_D_Pin|LED_C_Pin|LED_B_Pin|LED_A_Pin, GPIO_PIN_SET);

}
int led_cnt;
void Led_init(int led_sum)//»ÆÂÌµÆ½»ÌæÉÁ
{
	for(led_cnt=4;led_cnt<=led_sum;led_cnt++)
	{
		if(led_cnt%2==0)
		{
		HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_RESET);//GPIOE_PIN_11
		HAL_GPIO_WritePin(GREEN_GPIO_Port, GREEN_Pin, GPIO_PIN_RESET);//GPIOF_PIN_14
		HAL_Delay(50);

		}
		else
		{
		HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_SET);//GPIOE_PIN_7
		HAL_GPIO_WritePin(GREEN_GPIO_Port, GREEN_Pin, GPIO_PIN_SET);//GPIOF_PIN_14
		HAL_Delay(50);
			
		}
		HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_SET);//ºìµÆ
		HAL_GPIO_WritePin(GREEN_GPIO_Port, GREEN_Pin, GPIO_PIN_SET);//ÂÌµÆ
	}
		led_cnt=4;
	
	 HAL_GPIO_WritePin(LED_A_GPIO_Port, LED_A_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_B_GPIO_Port, LED_B_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_C_GPIO_Port, LED_C_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_D_GPIO_Port, LED_D_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_E_GPIO_Port, LED_E_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_F_GPIO_Port, LED_F_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_G_GPIO_Port, LED_G_Pin, GPIO_PIN_RESET);
	 HAL_GPIO_WritePin(LED_H_GPIO_Port, LED_H_Pin, GPIO_PIN_RESET);

}

