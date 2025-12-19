import express from 'express';
import { reviewController } from './controllers/review.controller';

const router = express.Router();

router.get('/api/products/:id/reviews', reviewController.getReviews);
router.post(
   '/api/products/:id/reviews/summarize',
   reviewController.summarizeReviews
);
export default router;
