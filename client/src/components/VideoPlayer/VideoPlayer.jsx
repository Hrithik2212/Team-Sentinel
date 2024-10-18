import React, { useEffect, useState } from 'react';


const VideoPlayer = ({videoSrc,loading,setVideoSrc,handleUpload}) => {
   


  return (
    <div className="relative flex h-full w-full items-center justify-center bg-[#111418] ">
      {videoSrc ? (
        <video
          src={videoSrc}
          autoPlay
          onEnded={()=>setVideoSrc(null)}
          muted
          className="h-full w-full object-cover"
        />
      ) : (
        <div className='w-full h-full relative '>
            {loading ?(
                <div className='z-0 absolute flex justify-center items-center cursor-wait  w-full h-full '>
                    <div className='bg-gray-50 absolute opacity-[0.3] w-full h-full'/>
                    <div className=' w-[40px] cursor-wait h-[40px] border-white opacity-1 z-10 border-l-0 border-t-0 animate-spin border-2 rounded-full bg-transparent'/>
                </div>
            ):(
                <div className='absolute w-full h-full text-white flex justify-center items-center  '>
                    <h4 className='w-full text-center mx-auto font-bold '>Upload</h4>
                    <input className='absolute cursor-pointer w-full h-full opacity-0'  type="file" accept="video/*" onChange={handleUpload}/>
                </div>
            )}
            <img className="h-full w-full" src='https://assets.timelinedaily.com/j/1203x902/2024/07/flipkart.jpg'/>
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;
