import React, { useRef, useState } from 'react'
import VideoPlayer from '../../../components/VideoPlayer/VideoPlayer';
import ProductCard from '../../../components/ProductCard/ProductCard';
import Products from '../Sample_Response'
import BASE_URL from '@/utils/baseApi';
const VideoSection = () => {
    const [videoSrc, setVideoSrc] = useState(null);
    const [loading,setLoading]=useState(false)
    const [productData,setProductData]=useState(null)
    const [show,setShow]=useState(null)
    const videoRef = useRef(null); 
  
  
  

    const handleUpload = async (event) => {
      if (!event.target.files[0]) return;
      setLoading(true)
  
      const formData = new FormData();
      formData.append('video', event.target.files[0]);
  
      try {
        const response = await fetch(BASE_URL+'upload-video/', {
          method: 'POST',
          body: formData,
        });
  
        const data = await response.json();
        setVideoSrc(`data:video/mp4;base64,${data.video}`);
        await new Promise((resolve) => setTimeout(resolve, 100));
        scheduleProductDisplay(data.products);
      } catch (error) {
        console.error('Error uploading video:', error);
      }
      finally{
        setLoading(false)
      }
    };
    const scheduleProductDisplay = (products) => {
      const videoElement = videoRef.current;
  
      if (!videoElement) {
        console.error('Video element is not available.');
        return;
      }
  
      videoElement.onloadedmetadata = () => {
        const videoDuration = videoElement.duration;
        const interval = videoDuration / products.length;
        setProductData([])
  
        products.forEach((product, index) => {
          setTimeout(() => {
            setProductData((prev) => [...prev, product]); // Append one product at a time
          }, interval * 1000 * (index+1)); // Schedule each product based on the interval
        });
      };
    };

  return (
    <React.Fragment>
    <div className='w-[70%] max-md:w-full max-md:h-[50vh]'>
        <VideoPlayer videoRef={videoRef} videoSrc={videoSrc} setVideoSrc={setVideoSrc} loading={loading} handleUpload={handleUpload}/>
      </div>
      <div className='shadow-lg pb-5 mx-auto w-full  h-full overflow-y-scroll md:w-[30%] bg-white'>
        {productData ? (
            <React.Fragment>
            {productData?.map((product,index) => (
                <div className='w-full cursor-pointer border-b' onClick={()=>setShow(index)} key={index}>
                    <ProductCard  product={product} />
                    {show===index && (
                        <table className=' mx-auto gap-2 m-5 w-[80%]  '>
                            <thead>
                                <tr>
                                    <th>Info</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {productData[index]?.track_id &&(
                                  <tr>
                                            <td>Tracker Id</td>
                                            <td>{productData[index]?.track_id}</td> 
                                  </tr>
                                )}
                                {productData[index]?.entities.product_name &&(
                                   <tr>
                                    <td>Class Name</td>
                                    <td>{productData[index]?.entities.product_name}</td> 
                                  </tr>
                                )}
                                {productData[index]?.entities.category &&(
                                   <tr>
                                      <td>Category</td>
                                      <td>{productData[index]?.entities.category}</td> 
                                    </tr>
                                )}
                                {productData[index]?.entities.brand_name &&(
                                   <tr>
                                      <td>Brand Name</td>
                                      <td>{productData[index]?.entities.brand_name}</td> 
                                    </tr>
                                )}
                                {productData[index]?.entities.brand_details &&(
                                   <tr>
                                      <td>Brand Details</td>
                                      <td>{productData[index]?.entities.brand_details}</td> 
                                    </tr>
                                )}
                                {productData[index]?.entities.pack_size &&(
                                    <tr>
                                              <td>Pack Size</td>
                                              <td>{productData[index]?.entities.pack_size}</td> 
                                    </tr>
                                )}
                                {productData[index]?.entities.expiry_date &&(
                                   <tr>
                                            <td>Expiry Date</td>
                                            <td>{productData[index]?.entities.expiry_date}</td> 
                                  </tr>
                                )}
                                {productData[index]?.entities.mrp &&(
                                   <tr>
                                            <td>MRP</td>
                                            <td>{productData[index]?.entities.mrp}</td> 
                                  </tr>
                                )}
                                {productData[index]?.entities.estimated_shelf_life &&(
                                   <tr>
                                            <td>Estimated Shelf Life</td>
                                            <td>{productData[index]?.entities.estimated_shelf_life}</td> 
                                  </tr>
                                )}
                                {productData[index]?.entities.state &&(
                                   <tr>
                                            <td>State</td>
                                            <td className={`${productData[index]?.entities.state ==="fresh" ?("bg-green-500"):("bg-red-500")}`}></td> 
                                  </tr>
                                )}

                                
                               
                                
                                
                                
                               
                                
                                
                            </tbody>
                            
                        </table>
                    )}
                </div>
            ))}
            
        </React.Fragment>
        ):(
            <React.Fragment>
                <h4 className='w-full h-full flex justify-center items-center px-5 py-2 text-[20px] text-center'>Welcome<br></br> to Team Sentinels at Flipkart Grid Hackathon!</h4>
            </React.Fragment>

        )}
      </div>

    </React.Fragment>
  )
}

export default VideoSection