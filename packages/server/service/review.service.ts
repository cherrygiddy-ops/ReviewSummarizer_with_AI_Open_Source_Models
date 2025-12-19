import reviewRepository from '../repositories/review.repository';
import { llmClient } from '../llm/client';

export const reviewService = {
   async summarizeReviews(productId: number): Promise<string> {
      const existingSummaries =
         await reviewRepository.getReviewSummary(productId);
      if (existingSummaries) return existingSummaries;

      const reviews = await reviewRepository.getReviews(productId, 10);
      const joinReviews = reviews.map((rev) => rev.content).join('\n\n');
      const summary = await llmClient.summarizeReviewsWithOllama(joinReviews);
      await reviewRepository.storeReviewSummary(productId, summary);
      return summary;
   },
};
