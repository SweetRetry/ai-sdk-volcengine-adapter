import {
  ImageModelV3,
  ImageModelV3File,
  SharedV3Warning,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { volcengineFailedResponseHandler } from '../chat/volcengine-error';
import { volcengineImageResponseSchema } from './volcengine-image-api';

export type VolcengineImageModelId =
  | 'doubao-seedream-4-5-251128'
  | 'doubao-seedream-4-0-250828'
  | (string & {});

export interface VolcengineImageConfig {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string>;
  fetch?: FetchFunction;
}

export class VolcengineImageModel implements ImageModelV3 {
  readonly specificationVersion = 'v3' as const;

  get maxImagesPerCall(): number {
    return 1;
  }

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: VolcengineImageModelId,
    private readonly config: VolcengineImageConfig,
  ) {}

  async doGenerate({
    prompt,
    files,
    size,
    aspectRatio,
    seed,
    providerOptions,
    headers,
    abortSignal,
  }: Parameters<ImageModelV3['doGenerate']>[0]): Promise<
    Awaited<ReturnType<ImageModelV3['doGenerate']>>
  > {
    const warnings: SharedV3Warning[] = [];

    // Handle unsupported options
    if (aspectRatio != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'aspectRatio',
      });
    }

    if (seed != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'seed',
      });
    }

    // Build request body - always use b64_json as AI SDK requires base64 data
    const { response_format: _, ...volcengineOptions } =
      (providerOptions.volcengine as Record<string, unknown>) ?? {};

    const body: Record<string, unknown> = {
      model: this.modelId,
      prompt,
      size: size ?? '2048x2048',
      ...volcengineOptions,
      response_format: 'b64_json',
    };

    // Handle reference images for image-to-image generation
    if (files && files.length > 0) {
      const imageUrls = files.map(file => this.convertFileToDataUrl(file));

      if (imageUrls.length === 1) {
        body.image = imageUrls[0];
      } else {
        body.image = imageUrls;
      }
    }

    const { value: response, responseHeaders } = await postJsonToApi({
      url: `${this.config.baseURL}/images/generations`,
      headers: combineHeaders(this.config.headers(), headers),
      body,
      failedResponseHandler: volcengineFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        volcengineImageResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    // Extract base64 images from response
    const images: string[] = response.data.map(item => {
      if (!item.b64_json) {
        throw new Error('No base64 image data in response');
      }
      return item.b64_json;
    });

    return {
      images,
      warnings,
      response: {
        timestamp: new Date(),
        modelId: response.model ?? this.modelId,
        headers: responseHeaders,
      },
      usage: response.usage
        ? {
            inputTokens: undefined,
            outputTokens: response.usage.output_tokens,
            totalTokens: response.usage.total_tokens,
          }
        : undefined,
    };
  }

  private convertFileToDataUrl(file: ImageModelV3File): string {
    if (file.type === 'url') {
      return file.url;
    }

    // file.type === "file"
    if (typeof file.data === 'string') {
      // base64 string - convert to data URL
      return `data:${file.mediaType};base64,${file.data}`;
    }

    // Uint8Array - convert to base64 data URL
    const base64 = btoa(String.fromCharCode(...file.data));
    return `data:${file.mediaType};base64,${base64}`;
  }
}
