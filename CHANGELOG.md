# Changelog

## 0.1.5

- 修复 Image Generation 返回 URL 时 AI SDK 解析失败的问题
- 强制使用 `b64_json` 格式返回图片数据（AI SDK 要求 base64 编码）

## 0.1.4

- 简化导出接口，移除内部实现细节的导出
- 代码格式统一（prettier）

## 0.1.1

- 增加 PDF 文件支持
- 更加标准的 AI SDK Provider 设计
- 增加 Image Generation（支持文生图、图生图、多图生图）

## 0.1.0

- 初始版本
- 支持火山引擎豆包大模型 Chat Completions API
- 支持流式和非流式文本生成
- 支持图片和视频输入
- 支持 Tool Calling（函数调用）
- 支持 Reasoning Content（思维链输出）
- 内置 Web Search 工具
