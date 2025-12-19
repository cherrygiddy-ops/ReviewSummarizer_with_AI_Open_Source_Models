import OpenAI from 'openai';
import { InferenceClient } from '@huggingface/inference';

const openAIClient = new OpenAI({
   apiKey: process.env.OPENAI_API_KEY,
});
const inferenceClient = new InferenceClient(process.env.HF_Token);
type GenerateTextOptions = {
   model?: string;
   temperature?: number;
   prompt: string;
   maxTokens?: number;
};

type GenerateTextResults = {
   id: string;
   text: string;
};

export const llmClient = {
   async generateText({
      prompt,
      maxTokens = 300,
      model = 'gpt-4.1',
      temperature = 0.2,
   }: GenerateTextOptions): Promise<GenerateTextResults> {
      const response = await openAIClient.responses.create({
         input: prompt,
         max_output_tokens: maxTokens,
         model,
         temperature,
      });
      return {
         id: response.id,
         text: response.output_text,
      };
   },
   async summarize(text: string) {
      const output = await inferenceClient.summarization({
         model: 'facebook/bart-large-cnn',
         inputs: text,
         provider: 'hf-inference',
      });
      return output.summary_text;
   },
};
