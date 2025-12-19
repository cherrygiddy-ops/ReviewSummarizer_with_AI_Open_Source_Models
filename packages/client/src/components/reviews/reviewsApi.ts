import axios from 'axios';

export type Review = {
   id: number;
   author: string;
   rating: number;
   content: string;
   createdAt: Date;
   productId: number;
};

export type ReviewResponse = {
   reviews: Review[];
   summary: string;
};

export type SummaryResponse = {
   summary: string;
};

export const reviewsApiClient = {
   fetchReviews(productId: number) {
      return axios
         .get<ReviewResponse>(`/api/products/${productId}/reviews`)
         .then((res) => res.data);
   },
   summarizeReviews(productId: number) {
      return axios
         .post<SummaryResponse>(`api/products/${productId}/reviews/summarize`)
         .then((res) => res.data);
   },
};
