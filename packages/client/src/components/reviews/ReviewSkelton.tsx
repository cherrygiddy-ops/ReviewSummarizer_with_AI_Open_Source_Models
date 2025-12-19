import Skeleton from 'react-loading-skeleton';

const ReviewSkelton = () => {
   return (
      <div>
         <Skeleton width={150}></Skeleton>
         <Skeleton width={100}></Skeleton>
         <Skeleton count={2}></Skeleton>
      </div>
   );
};

export default ReviewSkelton;
