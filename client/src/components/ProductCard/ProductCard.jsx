import React, { useEffect, useState } from 'react';

const ProductCard = ({ product}) =>{

  const [imageSrc, setImageSrc] = useState(null);

  useEffect(()=>{

    setImageSrc(product?.image_base64_url);

  },[product])


  return (
    <div className="flex gap-4 bg-white px-4 py-3">

              <img
                src={imageSrc}
                alt="Converted"
                className="bg-center bg-no-repeat aspect-video bg-cover rounded-lg h-[70px] w-fit"
              />
              <div className="flex flex-1 flex-col justify-center">
                <p className="text-[#111418] text-[0.9rem] font-medium leading-normal">{product?.class_name}</p>
                <p className="text-[#637588] text-sm font-normal leading-normal">{product?.entities?.brand_name}</p>
                <p className="text-[#637588] text-sm font-normal leading-normal">{product?.entities?.count}</p>
              </div>
    </div>
  
)};

export default ProductCard;
