import React from 'react';

const ProductCard = ({ name,count }) => (
  <div className="flex gap-4  w-[95%] mx-auto  px-4 py-5">
    <div className="flex  justify-between w-[90%] mx-auto">
      <p className="text-base font-medium">{name}</p>
      <p>{count}</p>
    </div>
  </div>
);

export default ProductCard;
