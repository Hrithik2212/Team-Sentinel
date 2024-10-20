import React, { useState } from 'react';

import ImageSection from './Image/ImageSection';
import VideoSection from './Video/VideoSection';


const HomePage = () => {
  const [choice,setChoice]=useState(false)
  

    return(
    
  <div className="min-h-full h-full  flex flex-col mt-5 ">
    <div className='w-[150px] flex justify-center  mx-auto mb-5 shadow-lg  bg-white '>
      <button onClick={()=>setChoice(false)} className={`border-r w-full h-full py-2  ${!choice && ("bg-gray-300")}`}>
        Video
      </button>
      <button onClick={()=>setChoice(true)} className={`border-r w-full h-full py-2  ${choice && ("bg-gray-300")}`}>
        Image
      </button>
    </div>
    {choice ? (
      <div className="md:px-10 flex  max-md:flex-col w-full h-[80%] max-md:h-full max-w-[100vw] mx-auto ">
          <ImageSection/>
      </div>
    ):(
      <div className=" md:px-10 flex  max-md:flex-col w-full h-[80%] max-md:h-full max-w-[100vw] mx-auto ">
          <VideoSection/>
      </div>
    )}
    
   
  </div>
    )
};

export default HomePage;
