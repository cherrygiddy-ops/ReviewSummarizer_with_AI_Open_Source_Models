import dayjs, { Dayjs } from 'dayjs'; // ✅ correct

import { PrismaClient, type Review } from '../generated/prisma';

const prisma = new PrismaClient();
const reviewRepository = {
   async getReviews(productId: number, limit?: number): Promise<Review[]> {
      return prisma.review.findMany({
         where: { productId },
         orderBy: { createdAt: 'desc' },
         take: limit,
      });
   },

   async storeReviewSummary(productId: number, summary: string) {
      const now = new Date();

      const expiresAt = dayjs().add(7, 'days').toDate(); // ✅

      const data = {
         content: summary,
         expiresAt: expiresAt,
         generatedAt: now,
         productId,
      };
      return prisma.summary.upsert({
         where: { productId },
         create: data,
         update: data,
      });
   },
   async getReviewSummary(productId: number): Promise<string | null> {
      const summarry = await prisma.summary.findFirst({
         where: {
            AND: [{ productId }, { expiresAt: { gt: new Date() } }],
         },
      });
      return summarry ? summarry.content : null;
   },
};

export default reviewRepository;
