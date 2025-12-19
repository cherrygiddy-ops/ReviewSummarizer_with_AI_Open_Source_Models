import OpenAI from 'openai';
import type { Review } from '../generated/prisma';
import reviewRepository from '../repositories/review.repository';
import { llmClient } from '../llm/client';
import template from '../prompts/summarize.reviews.txt';

export const reviewService = {
   async summarizeReviews(productId: number): Promise<string> {
      const existingSummaries =
         await reviewRepository.getReviewSummary(productId);
      if (existingSummaries) return existingSummaries;

      const reviews = await reviewRepository.getReviews(productId, 10);
      const joinReviews = reviews.map((rev) => rev.content).join('\n\n');
      const prompt = template.replace('{{reviews}}', joinReviews);
      const summary = await llmClient.summarize(joinReviews);
      await reviewRepository.storeReviewSummary(productId, summary);
      return summary;
   },
};
