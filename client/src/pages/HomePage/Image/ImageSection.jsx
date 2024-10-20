import ImageView from '@/components/ImageView/ImageView'
import React, { useState } from 'react'
import BASE_URL from '@/utils/baseApi';
const ImageSection = () => {
    const [productData,setProductData]=useState(null)
    const [imagePreviews, setImagePreviews] = useState(null);
    const [loading,setLoading]=useState(false)

    const sendImagesToBackend = async () => {
        console.log("called")
        setLoading(true)
        try {
          const response = await fetch(BASE_URL+'upload-video/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json', 
            },
            body: JSON.stringify({ images: imagePreviews }), 
          });
    
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
    
          const data = await response.json();
          setProductData(data?.products)
          console.log('Success:', data);
        } catch (error) {
          console.error('Error:', error); 
        }finally{
            setLoading(false)
        }
      };

    const handleImgUpload = (e) => {
        console.log("called")
      const files = Array.from(e.target.files); 
  
      const newImagePreviews = files.map((file) => {
        const reader = new FileReader();
        reader.readAsDataURL(file); 
  
        return new Promise((resolve) => {
          reader.onloadend = () => {
            resolve(reader.result); 
          };
        });
      });
      Promise.all(newImagePreviews).then((base64Images) => {
        setImagePreviews(base64Images);
      });
      sendImagesToBackend();
    }

    
  return (
    <React.Fragment>
    <div className='w-[70%] max-md:w-full max-md:h-[50vh]'>
        <ImageView imgSrc={imagePreviews} loading={loading} handleImgUpload={handleImgUpload}/>
      </div>
      <div className='shadow-lg pb-5 mx-auto w-full  h-full overflow-y-scroll md:w-[30%] bg-white'>
        {productData ? (
            <React.Fragment>
            <h4 className='w-full text-right px-5 py-2'>Count:- 15</h4>
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
                                <tr>
                                          <td>Tracker Id</td>
                                          <td>{productData[index]?.track_id}</td> 
                                </tr>
                                <tr>
                                          <td>Class Name</td>
                                          <td>{productData[index]?.entities.product_name}</td> 
                                </tr>
                                <tr>
                                          <td>Category</td>
                                          <td>{productData[index]?.entities.category}</td> 
                                </tr>
                                <tr>
                                          <td>Brand Name</td>
                                          <td>{productData[index]?.entities.brand_name}</td> 
                                </tr>
                                <tr>
                                          <td>Brand Details</td>
                                          <td>{productData[index]?.entities.brand_details}</td> 
                                </tr>
                                <tr>
                                          <td>Pack Size</td>
                                          <td>{productData[index]?.entities.pack_size}</td> 
                                </tr>
                                <tr>
                                          <td>Expiry Date</td>
                                          <td>{productData[index]?.entities.expiry_date}</td> 
                                </tr>
                                <tr>
                                          <td>MRP</td>
                                          <td>{productData[index]?.entities.mrp}</td> 
                                </tr>
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

export default ImageSection