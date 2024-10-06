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
  huart1.Instance = USART1;                         //串口1的基地址
  huart1.Init.BaudRate = 100000;                    //数据初始化波特率
  huart1.Init.WordLength = UART_WORDLENGTH_8B;      //数据初始化字长
  huart1.Init.StopBits = UART_STOPBITS_1;           //数据初始化停止位
  huart1.Init.Parity = UART_PARITY_EVEN;            //
  huart1.Init.Mode = UART_MODE_RX;               //数据初始化接收模式
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;       //无硬件流控
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;  //数据初始化过采样
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
	HAL_UART_Init(&huart2);//使能串口3
	
}
void MX_USART3_UART_Init(void)
{
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 9600;//WT901的默认波特率为9600  拿到的产品必须去上位机软件中确认波特率 
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart3.Init.OverSampling = UART_OVERSAMPLING_16;
	HAL_UART_Init(&huart3);//使能串口3
	HAL_UART_Receive_IT(&huart3, Re_buf_imu,sizeof(Re_buf_imu));//该函数会开启接收中断：标志位UART_IT_RXNE，并且设置接收缓冲以及接收缓冲接收最大数据量(使用回调函数处理中断需要调用该函数）
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

  HAL_UART_Init(&huart6);//使能串口6
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
	HAL_UART_Init(&huart7);//使能串口1
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
	HAL_UART_Init(&huart8);//使能串口1
}
void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{
  GPIO_InitTypeDef GPIO_InitStruct;
	  if(uartHandle->Instance==USART1)
  {
    __HAL_RCC_USART1_CLK_ENABLE();

    /*USART1 GPIO Configuration    //遥控器接口
    PB7     ------> USART1_RX
    PB6     ------> USART1_TX*/  
		//GPIO初始化结构体配置
    GPIO_InitStruct.Pin = GPIO_PIN_7|GPIO_PIN_6;     //GPIO的发送和接收引脚配置
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;          //GPIO初始化结构体片上外设输出
    GPIO_InitStruct.Pull = GPIO_PULLUP;              //GPIO上拉
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH; //GPIO速度配高速
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;       //GPIO端口复用为串口1
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

   /* Peripheral DMA init*/
  
    hdma_usart1_rx.Instance = DMA2_Stream5;                          //串口接收基地址：DMA2,数据流2
    hdma_usart1_rx.Init.Channel = DMA_CHANNEL_4;                     //通道选择4
		hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;            //方向选择从外设到存储器
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;                //外设非增量式模式
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;                    //存储器使能
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;   //外设数据大小
		hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;      //存储器数据大小
    hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;                           //外设流控模式、循环模式、普通模式
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_LOW;                 //DMA优先级，选低
    hdma_usart1_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;             //FIFO模式禁止
    HAL_DMA_Init(&hdma_usart1_rx);

    __HAL_LINKDMA(uartHandle,hdmarx,hdma_usart1_rx);                //链接DMA通道句柄和外设串口1句柄
	
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
		__HAL_UART_ENABLE_IT(&huart2,UART_IT_RXNE);		//开启接收中断	
		HAL_NVIC_SetPriority(USART2_IRQn,3,0);	//抢占优先级2，子优先级2
		HAL_NVIC_EnableIRQ(USART2_IRQn);		//使能USART3中断通道
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
		__HAL_UART_ENABLE_IT(&huart3,UART_IT_RXNE);		//开启接收中断	
		HAL_NVIC_SetPriority(USART3_IRQn,2,2);	//抢占优先级2，子优先级2
		HAL_NVIC_EnableIRQ(USART3_IRQn);		//使能USART3中断通道
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
		__HAL_UART_ENABLE_IT(&huart6,UART_IT_RXNE);		//开启接收中断	
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
	HAL_UART_IRQHandler(&huart6);	//调用HAL库中断处理公用函数
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
	if(huart->Instance==USART3)//如果是串口3
	{
			for(int i=0;i<=(USART3_REC_LEN-1);i++)  //筛选数据
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

	if(huart->Instance==USART6)//如果是串口6
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
  rc->ch4 -= (0x016C);//   0x7E+0xEE = 0x016C   这个控制无人机上升下降  无需负值

  rc->sw1 = ((buff[5] >> 4) & 0x000C) >> 2;
  rc->sw2 = (buff[5] >> 4) & 0x0003;
  
	rc->ch1 =	abs(rc->ch1) > 5 ? rc->ch1 : 0;
	rc->ch2 =	abs(rc->ch2) > 5 ? rc->ch2 : 0;
	rc->ch3 =	abs(rc->ch3) > 5 ? rc->ch3 : 0;
	rc->ch4 =	abs(rc->ch4) > 5 ? rc->ch4 : 0;
 
	}
	__HAL_UART_CLEAR_PEFLAG(&huart1);
}
