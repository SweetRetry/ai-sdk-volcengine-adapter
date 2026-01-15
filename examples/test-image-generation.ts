import { config } from 'dotenv';
import { createVolcengine } from 'ai-sdk-volcengine-adapter';
import { generateImage } from 'ai';
import { writeFileSync, appendFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

// 加载 .env.local 文件
config({ path: '.env.local' });

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, 'output');
mkdirSync(OUTPUT_DIR, { recursive: true });
const OUTPUT_FILE = join(OUTPUT_DIR, 'test-image-generation-output.txt');

const volcengine = createVolcengine({
  apiKey: process.env.ARK_API_KEY,
});

function log(message: string) {
  console.log(message);
  appendFileSync(OUTPUT_FILE, message + '\n');
}

async function testImageGeneration() {
  log('=== 测试 Image Generation (generateImage API) ===\n');

  try {
    const result = await generateImage({
      model: volcengine.imageModel('doubao-seedream-4-5-251128'),
      prompt: '一只可爱的卡通猫咪，背景是蓝天白云',
      size: '2048x2048',
    });

    log('--- Response ---');

    log('\n[Generated Images]:');
    log(`Number of images: ${result.images.length}`);

    for (let i = 0; i < result.images.length; i++) {
      const image = result.images[i];
      log(`\nImage ${i + 1}:`);
      log(`  - Has base64 data: ${image.base64.length > 0}`);
      log(`  - Base64 length: ${image.base64.length}`);

      // 保存图片到文件
      const imageFileName = join(OUTPUT_DIR, `generated-image-${i + 1}.png`);
      const imageBuffer = Buffer.from(image.base64, 'base64');
      writeFileSync(imageFileName, imageBuffer);
      log(`  - Saved to: ${imageFileName}`);
    }

    log('\n--- Warnings ---');
    if (result.warnings && result.warnings.length > 0) {
      for (const warning of result.warnings) {
        log(`  - ${JSON.stringify(warning)}`);
      }
    } else {
      log('No warnings');
    }

    log('\n--- Response Metadata ---');
    log(`Model ID: ${result.providerMetadata.modelId}`);
    log(`Timestamp: ${result.providerMetadata.timestamp}`);

    if (result.usage) {
      log('\n--- Usage ---');
      log(JSON.stringify(result.usage, null, 2));
    }
  } catch (error) {
    log('Error: ' + String(error));
    if (error instanceof Error) {
      log('Stack: ' + error.stack);
    }
  }
}

async function main() {
  // 初始化输出文件
  writeFileSync(
    OUTPUT_FILE,
    `Image Generation Tool Test - ${new Date().toISOString()}\n${'='.repeat(
      60,
    )}\n\n`,
  );

  // 检查 API Key
  if (!process.env.ARK_API_KEY) {
    log('Error: ARK_API_KEY environment variable is not set');
    log('Please set your Volcengine API key:');
    log('  export ARK_API_KEY=your-api-key');
    process.exit(1);
  }

  log('Starting image generation tool test...\n');

  await testImageGeneration();

  log('\n\nTest completed!');
  log(`\nOutput saved to: ${OUTPUT_FILE}`);
}

main();
