#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "model_data.h"

static const char *TAG = "detector";

// 预处理：Resize(256) -> CenterCrop(224) -> Normalize
// 注意：实际部署时需要接入摄像头驱动（如 esp32-camera）
// 并实现 ONNX Runtime 或 ESP-DL 推理

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "螺丝/螺母 检测器 v1.0");
    ESP_LOGI(TAG, "模型: MobileNetV2 INT8");
    ESP_LOGI(TAG, "模型大小: %u bytes", model_onnx_len);
    ESP_LOGI(TAG, "输入尺寸: %dx%dx%d",
             MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS);
    ESP_LOGI(TAG, "类别数: %d (0=screw, 1=washer)", MODEL_NUM_CLASSES);
    ESP_LOGI(TAG, "置信度阈值: %.2f", CONFIG_MODEL_THRESHOLD);

    // TODO:
    // 1. 初始化摄像头 (esp_camera_init)
    // 2. 初始化 ONNX Runtime / ESP-DL
    // 3. 循环采集->预处理->推理->显示结果

    while (1) {
        // 伪代码:
        // camera_fb_t *fb = esp_camera_fb_get();
        // 预处理: resize + center_crop + normalize
        // 推理: onnx_model_run(model_data, input_tensor, output)
        // softmax + argmax
        // 显示结果
        ESP_LOGI(TAG, "等待摄像头初始化...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
