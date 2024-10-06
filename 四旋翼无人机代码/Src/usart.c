#include "usart.h"
#include "gpio.h"
#include "mytype.h"
#include "bsp_usart.h"

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3; 
UART_HandleTypeDef huart6;
UART_HandleTypeDef huart7;
UART_HandleTypeDef huart8;
DMA_HandleTypeDef hdma_usart1_rx;
RC_Type_STRUCT Dbus;
u8	RC_Buff[RC_BUF_LEN];

unsigned char Re_buf_imu[USART3_REC_LEN];
unsigned char IMU_acc[11];	 
unsigned char IMU_angle_speed[11];
unsigned char IMU_angle[11];

unsigned char buf_pc[16];
/* USART1 init function */
void MX_USART1_UART_Init(void)      
{
  huart1.Instance = USART1;                         //����1�Ļ���ַ
  huart1.Init.BaudRate = 100000;                    //���ݳ�ʼ��������
  huart1.Init.WordLength = UART_WORDLENGTH_8B;      //���ݳ�ʼ���ֳ�
  huart1.Init.StopBits = UART_STOPBITS_1;           //���ݳ�ʼ��ֹͣλ
  huart1.Init.Parity = UART_PARITY_EVEN;            //
  huart1.Init.Mode = UART_MODE_RX;               //���ݳ�ʼ������ģʽ
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;       //��Ӳ������
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;  //���ݳ�ʼ��������
  HAL_UART_Init(&huart1);
	
}
void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart2.Init.OverSampling = UART_OVERSAMPLING_16;
	HAL_UART_Init(&huart2);//ʹ�ܴ���3
	
}
void MX_USART3_UART_Init(void)
{
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 9600;//WT901��Ĭ�ϲ�����Ϊ9600  �õ��Ĳ�Ʒ����ȥ��λ�������ȷ�ϲ����� 
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart3.Init.OverSampling = UART_OVERSAMPLING_16;
	HAL_UART_Init(&huart3);//ʹ�ܴ���3
	HAL_UART_Receive_IT(&huart3, Re_buf_imu,sizeof(Re_buf_imu));//�ú����Ὺ�������жϣ���־λUART_IT_RXNE���������ý��ջ����Լ����ջ���������������(ʹ�ûص����������ж���Ҫ���øú�����
}
void MX_USART6_UART_Init(void)
{
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 115200;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;

  HAL_UART_Init(&huart6);//ʹ�ܴ���6
	//HAL_UART_Receive_IT(&huart6,(u8 *)aRxBuffer, RXBUFFERSIZE);
}
void MX_USART7_UART_Init(void)
{
  huart7.Instance = UART7;
  huart7.Init.BaudRate = 115200;
  huart7.Init.WordLength = UART_WORDLENGTH_8B;
  huart7.Init.StopBits = UART_STOPBITS_1;
  huart7.Init.Parity = UART_PARITY_EVEN;
  huart7.Init.Mode = UART_MODE_TX_RX;
  huart7.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart7.Init.OverSampling = UART_OVERSAMPLING_16;
	HAL_UART_Init(&huart7);//ʹ�ܴ���1
}
void MX_USART8_UART_Init(void)
{
  huart8.Instance = UART8;
  huart8.Init.BaudRate = 115200;
  huart8.Init.WordLength = UART_WORDLENGTH_8B;
  huart8.Init.StopBits = UART_STOPBITS_1;
  huart8.Init.Parity = UART_PARITY_NONE;
  huart8.Init.Mode = UART_MODE_TX_RX;
  huart8.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart8.Init.OverSampling = UART_OVERSAMPLING_16;
	HAL_UART_Init(&huart8);//ʹ�ܴ���1
}
void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{
  GPIO_InitTypeDef GPIO_InitStruct;
	  if(uartHandle->Instance==USART1)
  {
    __HAL_RCC_USART1_CLK_ENABLE();

    /*USART1 GPIO Configuration    //ң�����ӿ�
    PB7     ------> USART1_RX
    PB6     ------> USART1_TX*/  
		//GPIO��ʼ���ṹ������
    GPIO_InitStruct.Pin = GPIO_PIN_7|GPIO_PIN_6;     //GPIO�ķ��ͺͽ�����������
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;          //GPIO��ʼ���ṹ��Ƭ���������
    GPIO_InitStruct.Pull = GPIO_PULLUP;              //GPIO����
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH; //GPIO�ٶ������
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;       //GPIO�˿ڸ���Ϊ����1
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

   /* Peripheral DMA init*/
  
    hdma_usart1_rx.Instance = DMA2_Stream5;                          //���ڽ��ջ���ַ��DMA2,������2
    hdma_usart1_rx.Init.Channel = DMA_CHANNEL_4;                     //ͨ��ѡ��4
		hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;            //����ѡ������赽�洢��
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;                //���������ʽģʽ
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;                    //�洢��ʹ��
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;   //�������ݴ�С
		hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;      //�洢�����ݴ�С
    hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;                           //��������ģʽ��ѭ��ģʽ����ͨģʽ
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_LOW;                 //DMA���ȼ���ѡ��
    hdma_usart1_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;             //FIFOģʽ��ֹ
    HAL_DMA_Init(&hdma_usart1_rx);

    __HAL_LINKDMA(uartHandle,hdmarx,hdma_usart1_rx);                //����DMAͨ����������贮��1���
	
    /* Peripheral interrupt init */
    HAL_NVIC_SetPriority(USART1_IRQn, 2, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
		
		HAL_DMA_Start_IT(&hdma_usart1_rx,(uint32_t)huart1.Instance->DR,(uint32_t)RC_Buff,RC_BUF_LEN);
		huart1.Instance->CR3 |= USART_CR3_DMAR;
		__HAL_UART_ENABLE_IT(&huart1,UART_IT_IDLE);
		HAL_UART_Receive_DMA(&huart1,RC_Buff,RC_BUF_LEN);
		
   }
  if(uartHandle->Instance==USART2)
  {		 
		__HAL_RCC_GPIOD_CLK_ENABLE();
		__HAL_RCC_USART2_CLK_ENABLE();
    /**USART2 GPIO Configuration    
    PD6     ------> USART2_RX
    PD5     ------> USART2_TX 
    */
    GPIO_InitStruct.Pin = GPIO_PIN_5|GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
		__HAL_UART_ENABLE_IT(&huart2,UART_IT_RXNE);		//���������ж�	
		HAL_NVIC_SetPriority(USART2_IRQn,3,0);	//��ռ���ȼ�2�������ȼ�2
		HAL_NVIC_EnableIRQ(USART2_IRQn);		//ʹ��USART3�ж�ͨ��
	}
	if(uartHandle->Instance==USART3)
  {		 
		__HAL_RCC_GPIOD_CLK_ENABLE();
		__HAL_RCC_USART3_CLK_ENABLE();
    /**USART3 GPIO Configuration    
    PD9     ------> USART3_RX
    PD8     ------> USART3_TX 
    */
    GPIO_InitStruct.Pin = GPIO_PIN_9|GPIO_PIN_8;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
		__HAL_UART_ENABLE_IT(&huart3,UART_IT_RXNE);		//���������ж�	
		HAL_NVIC_SetPriority(USART3_IRQn,2,2);	//��ռ���ȼ�2�������ȼ�2
		HAL_NVIC_EnableIRQ(USART3_IRQn);		//ʹ��USART3�ж�ͨ��
	}
  else if(uartHandle->Instance==USART6)
  {
    __HAL_RCC_GPIOG_CLK_ENABLE();
		__HAL_RCC_USART6_CLK_ENABLE();
    /**USART6 GPIO Configuration    
    PG14     ------> USART6_TX
    PG9     ------> USART6_RX */
    GPIO_InitStruct.Pin = GPIO_PIN_14|GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF8_USART6;
    HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);
		__HAL_UART_ENABLE_IT(&huart6,UART_IT_RXNE);		//���������ж�	
		HAL_NVIC_EnableIRQ(USART6_IRQn);
		HAL_NVIC_SetPriority(USART6_IRQn, 2, 0);

  }
	 else if(uartHandle->Instance==UART7)
  {
    __HAL_RCC_UART7_CLK_ENABLE();
		__HAL_RCC_GPIOE_CLK_ENABLE();
    /**USART7 GPIO Configuration    
    PE7     ------> USART7_RX
    PB8     ------> USART7_TX */
    GPIO_InitStruct.Pin = GPIO_PIN_7|GPIO_PIN_8;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF8_UART7;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
  
		HAL_NVIC_EnableIRQ(UART7_IRQn);
		HAL_NVIC_SetPriority(UART7_IRQn, 2, 2);
  }
  else if(uartHandle->Instance==UART8)
  {
    __HAL_RCC_UART8_CLK_ENABLE();
		__HAL_RCC_GPIOE_CLK_ENABLE();
    /**USART8 GPIO Configuration    
    PE0     ------> USART8_RX
    PE1     ------> USART8_TX 
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF8_UART8;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
		  
		HAL_NVIC_EnableIRQ(UART8_IRQn);
		HAL_NVIC_SetPriority(UART8_IRQn, 2, 3);
  }

}
void USART1_IRQHandler(void)
{
if(__HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE) != RESET){
			__HAL_UART_CLEAR_IDLEFLAG(&huart1);		
			HAL_UART_DMAStop(&huart1);
			HAL_UART_Receive_DMA(&huart1,RC_Buff,RC_BUF_LEN);
		}
}

void USART3_IRQHandler(void)
{
	HAL_UART_IRQHandler(&huart3);
}
void USART6_IRQHandler(void)
{
	HAL_UART_IRQHandler(&huart6);	//����HAL���жϴ����ú���
}
void USART7_IRQHandler(void)
{ 
  HAL_UART_IRQHandler(&huart7);
}

void USART8_IRQHandler(void)
{
  HAL_UART_IRQHandler(&huart8);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{	
	if(huart->Instance==USART3)//����Ǵ���3
	{
			for(int i=0;i<=(USART3_REC_LEN-1);i++)  //ɸѡ����
			{
				if(Re_buf_imu[i] == 0x55 && Re_buf_imu[i+1] == 0x51)
				{		
					for(int j=0;j<=10;j++)
					IMU_acc[j] = Re_buf_imu[i+j];
				}
				if(Re_buf_imu[i] == 0x55 && Re_buf_imu[i+1] == 0x52)
				{		
					for(int j=0;j<=10;j++)
					IMU_angle_speed[j] = Re_buf_imu[i+j];
				}
				if(Re_buf_imu[i] == 0x55 && Re_buf_imu[i+1] == 0x53)
				{		
					for(int j=0;j<=10;j++)
					IMU_angle[j] = Re_buf_imu[i+j];
				}
			}

	}

	if(huart->Instance==USART6)//����Ǵ���6
	{

	}	

}
void DBUS_Decode(RC_Type_STRUCT *rc, uint8_t *buff)
{

	if(buff[12] < 0x02 && buff[13] < 0x02)
	{
  rc->ch1 = (buff[0] | buff[1] << 8) & 0x07FF;
  rc->ch1 -= 1024;
  rc->ch2 = (buff[1] >> 3 | buff[2] << 5) & 0x07FF;
  rc->ch2 -= 1024;
  rc->ch3 = (buff[2] >> 6 | buff[3] << 2 | buff[4] << 10) & 0x07FF;
  rc->ch3 -= 1024;
  rc->ch4 = (buff[4] >> 1 | buff[5] << 7) & 0x07FF;
  rc->ch4 -= (0x016C);//   0x7E+0xEE = 0x016C   ����������˻������½�  ���踺ֵ

  rc->sw1 = ((buff[5] >> 4) & 0x000C) >> 2;
  rc->sw2 = (buff[5] >> 4) & 0x0003;
  
	rc->ch1 =	abs(rc->ch1) > 5 ? rc->ch1 : 0;
	rc->ch2 =	abs(rc->ch2) > 5 ? rc->ch2 : 0;
	rc->ch3 =	abs(rc->ch3) > 5 ? rc->ch3 : 0;
	rc->ch4 =	abs(rc->ch4) > 5 ? rc->ch4 : 0;
 
	}
	__HAL_UART_CLEAR_PEFLAG(&huart1);
}
