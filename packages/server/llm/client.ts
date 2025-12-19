import OpenAI from 'openai';
import { InferenceClient } from '@huggingface/inference';
import summarizeprompt from '../llm/prompts/summarize.reviews.txt';

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
   async summarizeReviews(reviews: string) {
      const chatCompletion = await inferenceClient.chatCompletion({
         model: 'meta-llama/Llama-3.1-8B-Instruct',
         messages: [
            {
               role: 'system',
               content: summarizeprompt,
            },
            {
               role: 'user',
               content: reviews,
            },
         ],
      });

      // Depending on the client, the response is usually in completion.choices[0].message.content
      return chatCompletion.choices[0]?.message.content ||'';
   },
};
