import StartRatings from './StartRatings';
import { HiSparkles } from 'react-icons/hi2';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Button } from '../ui/button';
import ReviewSkelton from './ReviewSkelton';
import {
   reviewsApiClient,
   type ReviewResponse,
   type SummaryResponse,
} from './reviewsApi';
interface Props {
   productId: number;
}

const ReviewList = ({ productId }: Props) => {
   const summaryMutations = useMutation<SummaryResponse>({
      mutationFn: () => reviewsApiClient.summarizeReviews(productId),
      onSuccess: () => {
         reviewQuery.refetch();
      },
   });
   const reviewQuery = useQuery<ReviewResponse>({
      queryKey: ['reviews', productId],
      queryFn: () => reviewsApiClient.fetchReviews(productId),
   });
   if (reviewQuery.isLoading)
      return (
         <div className="flex flex-col gap-5 ">
            {[1, 2, 3, 4].map((rev) => (
               <ReviewSkelton key={rev}></ReviewSkelton>
            ))}
         </div>
      );
   if (reviewQuery.isError)
      return <p className="text-red-500">{reviewQuery.error.message}</p>;

   if (!reviewQuery.data?.reviews.length) return null;
   const currentSummary = reviewQuery.data?.summary;
   return (
      <div>
         <div className="mb-5">
            {currentSummary ? (
               <p>{currentSummary}</p>
            ) : (
               <div>
                  <Button
                     onClick={() => summaryMutations.mutate()}
                     className="cursor-pointer"
                     disabled={summaryMutations.isPending}
                  >
                     <HiSparkles></HiSparkles>Summarize
                  </Button>
                  {summaryMutations.isPending && (
                     <div className="py-3">
                        <ReviewSkelton></ReviewSkelton>
                     </div>
                  )}
                  {summaryMutations.isError && (
                     <p className="text-red-500">
                        Could not summarize review .Try again
                     </p>
                  )}
               </div>
            )}
         </div>
         <div className="flex flex-col gap-5 ">
            {reviewQuery.data?.reviews.map((review) => (
               <div key={review.id}>
                  <div className="font-semibold">{review.author}</div>
                  <div>
                     Ratings:<StartRatings value={review.rating}></StartRatings>
                  </div>
                  <div className="py-2">{review.content}</div>
               </div>
            ))}
         </div>
      </div>
   );
};

export default ReviewList;
