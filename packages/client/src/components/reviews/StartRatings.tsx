import { FaRegStar, FaStar } from 'react-icons/fa';
type Props = {
   value: number;
};
const StartRatings = ({ value }: Props) => {
   const placeholders = [1, 2, 3, 4, 5];
   return (
      <div className="flex gap-1 text-yellow-500">
         {placeholders.map((p) =>
            p <= value ? (
               <FaStar key={p}></FaStar>
            ) : (
               <FaRegStar key={p}></FaRegStar>
            )
         )}
      </div>
   );
};

export default StartRatings;
