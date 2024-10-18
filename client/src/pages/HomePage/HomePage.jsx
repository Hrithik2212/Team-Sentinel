import React, { useState } from 'react';
import BASE_URL from '@/utils/baseApi';
import VideoPlayer from '../../components/VideoPlayer/VideoPlayer';
import ProductCard from '../../components/ProductCard/ProductCard';

const products = [
  {
    image: 'https://cdn.usegalileo.ai/stability/2c86ce59-e20d-4077-9454-11ff7ee7123e.png',
    name: 'Monogram Empreinte Artsy MM',
    brand: 'Louis Vuitton',
    price: 'Handbag • $2500',
  },
  {
    image: 'https://cdn.usegalileo.ai/replicate/f0cfe11b-fcb8-40d9-b50c-6519d0df5ce2.png',
    name: 'Midi Dress',
    brand: 'Zara',
    price: 'Dress • $250',
  },
];

const HomePage = () => {
    const [videoSrc, setVideoSrc] = useState(null);
    const [loading,setLoading]=useState(false)
    const [productData,setProductData]=useState(null)
    const [show,setShow]=useState(null)
  
  
  

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
        setProductData(data.products)
      } catch (error) {
        console.error('Error uploading video:', error);
      }
      finally{
        setLoading(false)
      }
    };


    return(
    
  <div className="min-h-full h-full  flex flex-col mt-5 ">
    <div className=" md:px-10 flex max-md:flex-col w-full h-[80%] max-md:h-full max-w-[90vw] mx-auto ">
      <div className='w-[70%] max-md:w-full max-md:h-[50vh]'>
        <VideoPlayer videoSrc={videoSrc} setVideoSrc={setVideoSrc} loading={loading} handleUpload={handleUpload}/>
      </div>
      <div className='shadow-lg pb-5 mx-auto w-full  h-full overflow-y-scroll md:w-[30%] bg-white'>
        {productData ? (
            <React.Fragment>
            <h4 className='w-full text-right px-5 py-2'>Count:- 15</h4>
            {productData?.map((product,index) => (
                <div className='w-full cursor-pointer border-b' onClick={()=>setShow(index)} key={index}>
                    <ProductCard   {...product} />
                    {show===index && (
                        <table className=' mx-auto gap-2 m-5 w-[80%]  '>
                            <thead>
                                <tr>
                                    <th>Info</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.keys(product)?.map((item,key)=>(
                                    <tr key={key}>
                                        <td>{item}</td>
                                        <td>{product[item]}</td> 
                                    </tr>
                                ))}
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
    </div>
  </div>
    )
};

export default HomePage;
