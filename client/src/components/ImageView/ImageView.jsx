import React from 'react';

const ImageView = ({ imgSrc, loading, handleImgUpload }) => {
  return (
    <div className="relative flex h-full border w-full items-center justify-center bg-[#111418] overflow-hidden">
      {imgSrc && imgSrc.length > 0 ? (
        <div className="w-full h-full relative flex gap-2">
      
              <input className='absolute cursor-pointer w-full h-full opacity-0' type="file" accept="image/*" multiple onChange={handleImgUpload} />
              {imgSrc.map((src, index) => (
                  <img
                    key={index}
                    src={src}
                    alt={`Uploaded Preview ${index + 1}`}
                    className="w-1/2 h-full rounded-lg" 
                  />
              ))}
        </div>
       
      ) : (
        <div className='w-full h-full relative'>
          {loading ? (
            <div className='z-0 absolute flex justify-center items-center cursor-wait w-full h-full'>
              <div className='bg-gray-50 absolute opacity-[0.3] w-full h-full' />
              <div className='w-[40px] cursor-wait h-[40px] border-white opacity-1 z-10 border-l-0 border-t-0 animate-spin border-2 rounded-full bg-transparent' />
            </div>
          ) : (
            <div className='absolute w-full h-full text-white flex justify-center items-center'>
              <h4 className='w-full text-center mx-auto font-bold'>Upload Image</h4>
              <input className='absolute cursor-pointer w-full h-full opacity-0' type="file" accept="image/*" multiple onChange={handleImgUpload} />
            </div>
          )}
          <img className="h-full w-full" src='https://assets.timelinedaily.com/j/1203x902/2024/07/flipkart.jpg' />
        </div>
      )}
    </div>
  );
};

export default ImageView;
