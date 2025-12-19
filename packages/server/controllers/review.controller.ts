import type { Request, Response } from 'express';
import { reviewService } from '../service/review.service';
import { productRepository } from '../repositories/product.repository';
import reviewRepository from '../repositories/review.repository';

export const reviewController = {
   async getReviews(req: Request, res: Response) {
      const productId = Number(req.params.id);
      if (isNaN(productId)) {
         res.status(400).json({ error: 'invalid product id passed' });
         return;
      }
      const product = await productRepository.getProduct(productId);
      if (!product) {
         res.status(404).json({ error: 'product doesnt exists' });
         return;
      }
      const reviews = await reviewRepository.getReviews(productId);
      const summary = await reviewRepository.getReviewSummary(productId);
      res.json({
         summary,
         reviews,
      });
   },

   async summarizeReviews(req: Request, res: Response) {
      var productId = Number(req.params.id);
      if (isNaN(productId)) {
         res.status(404).json({ error: 'invalid product id' });
      }
      const product = await productRepository.getProduct(productId);
      if (!product) {
         res.status(400).json({ error: 'product doesnt exists' });
         return;
      }
      const reviews = await reviewRepository.getReviews(productId, 1);
      if (reviews.length < 1) {
         res.status(400).json({ error: 'no reviews to sumarrieze' });
         return;
      }
      const summary = await reviewService.summarizeReviews(productId);

      res.json(summary);
   },
};
