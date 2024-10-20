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

  
    const sendImagesToBackend = async (base64Images) => {
      setLoading(true);
      try {
        const response = await fetch(BASE_URL + '/upload-image/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ images: base64Images }),
        });
    
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
    
        const data = await response.json();
        setProductData(data);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };
    
    const handleImgUpload = (e) => {
      console.log('called');
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
        setImagePreviews(base64Images)
        sendImagesToBackend(base64Images);
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
                             {productData?.entities.product_name &&(
                                <tr>
                                 <td>Class Name</td>
                                 <td>{productData?.entities.product_name}</td> 
                               </tr>
                             )}
                             {productData?.entities.category &&(
                                <tr>
                                   <td>Category</td>
                                   <td>{productData?.entities.category}</td> 
                                 </tr>
                             )}
                             {productData?.entities.brand_name &&(
                                <tr>
                                   <td>Brand Name</td>
                                   <td>{productData?.entities.brand_name}</td> 
                                 </tr>
                             )}
                             {productData?.entities.brand_details &&(
                                <tr>
                                   <td>Brand Details</td>
                                   <td>{productData?.entities.brand_details}</td> 
                                 </tr>
                             )}
                             {productData?.entities.pack_size &&(
                                 <tr>
                                           <td>Pack Size</td>
                                           <td>{productData?.entities.pack_size}</td> 
                                 </tr>
                             )}
                             {productData?.entities.expiry_date &&(
                                <tr>
                                         <td>Expiry Date</td>
                                         <td>{productData?.entities.expiry_date}</td> 
                               </tr>
                             )}
                             {productData?.entities.mrp &&(
                                <tr>
                                         <td>MRP</td>
                                         <td>{productData?.entities.mrp}</td> 
                               </tr>
                             )}
                             {productData?.entities.estimated_shelf_life &&(
                                <tr>
                                         <td>Estimated Shelf Life</td>
                                         <td>{productData?.entities.estimated_shelf_life}</td> 
                               </tr>
                             )}
                             {productData?.entities.state &&(
                                <tr>
                                         <td>State</td>
                                         <td className={`${productData?.entities.state ==="fresh" ?("bg-green-500"):("bg-red-500")}`}></td> 
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