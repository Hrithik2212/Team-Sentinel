import ImageView from '@/components/ImageView/ImageView'
import React, { useEffect, useState } from 'react'
import BASE_URL from '@/utils/baseApi';
import Products from '../Sample_Response'
import ProductCard from '@/components/ProductCard/ProductCard';
const ImageSection = () => {
    const [productData,setProductData]=useState(null)
    const [imagePreviews, setImagePreviews] = useState(null);
    const [loading,setLoading]=useState(false)
    const [show,setShow]=useState(null)

  
    const sendImagesToBackend = async (files) => {
      const formData = new FormData();
      files.forEach((file) => formData.append('image', file)); // Append all files
  
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/upload-image/', {
          method: 'POST',
          body: formData,
        });
  
        if (!response.ok) {
          throw new Error('Failed to upload images');
        }
  
        const data = await response.json();
        setProductData(data); 
      } catch (error) {
        console.error('Error uploading images:', error);
      } finally {
        setLoading(false);
      }
    };
    
    const handleImgUpload = (e) => {
      const files = Array.from(e.target.files);
  
      const previews = files.map((file) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
  
        return new Promise((resolve) => {
          reader.onloadend = () => resolve(reader.result);
        });
      });
  
      Promise.all(previews).then((base64Previews) => {
        setImagePreviews(base64Previews);
        sendImagesToBackend(files);
      });
    };
    

    
  return (
    <React.Fragment>
    <div className='w-[70%] max-md:w-full max-md:h-[50vh]'>
        <ImageView imgSrc={imagePreviews} loading={loading} handleImgUpload={handleImgUpload}/>
      </div>
      <div className='shadow-lg pb-5 mx-auto w-full  h-full overflow-y-scroll md:w-[30%] bg-white'>
        {productData ? (
            <React.Fragment>
          
        
                <div className='w-full cursor-pointer border-b' onClick={()=>setShow(!show)} >
                    <ProductCard  product={productData} />
                    {show && (
                         <table className=' mx-auto gap-2 m-5 w-[80%]  '>
                         <thead>
                             <tr>
                                 <th>Info</th>
                                 <th>Value</th>
                             </tr>
                         </thead>
                         <tbody>
                             {productData?.track_id &&(
                               <tr>
                                         <td>Tracker Id</td>
                                         <td>{productData?.track_id}</td> 
                               </tr>
                             )}
                             {productData?.product_name &&(
                                <tr>
                                 <td>Class Name</td>
                                 <td>{productData?.product_name}</td> 
                               </tr>
                             )}
                             {productData?.category &&(
                                <tr>
                                   <td>Category</td>
                                   <td>{productData?.category}</td> 
                                 </tr>
                             )}
                             {productData?.brand_name &&(
                                <tr>
                                   <td>Brand Name</td>
                                   <td>{productData?.brand_name}</td> 
                                 </tr>
                             )}
                             {productData?.brand_details &&(
                                <tr>
                                   <td>Brand Details</td>
                                   <td>{productData?.brand_details}</td> 
                                 </tr>
                             )}
                             {productData?.pack_size &&(
                                 <tr>
                                           <td>Pack Size</td>
                                           <td>{productData?.pack_size}</td> 
                                 </tr>
                             )}
                             {productData?.expiry_date &&(
                                <tr>
                                         <td>Expiry Date</td>
                                         <td>{productData?.expiry_date}</td> 
                               </tr>
                             )}
                             {productData?.mrp &&(
                                <tr>
                                         <td>MRP</td>
                                         <td>{productData?.mrp}</td> 
                               </tr>
                             )}
                             {productData?.estimated_shelf_life &&(
                                <tr>
                                         <td>Estimated Shelf Life</td>
                                         <td>{productData?.estimated_shelf_life}</td> 
                               </tr>
                             )}
                             {productData?.state &&(
                                <tr>
                                         <td>State</td>
                                         <td className={`${productData?.state ==="fresh" ?("bg-green-700 fonfont-t-bold"):("bg-red-500 bold")}`}></td> 
                               </tr>
                             )}

                             
                            
                             
                             
                             
                            
                             
                             
                         </tbody>
                         
                     </table>
                    )}
                </div>
            
            
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