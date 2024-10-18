import React, { useContext } from 'react'
import AuthContext from '../../context/AuthContext'
import {Link} from 'react-router-dom'
const Login = () => {
  let {loginUser} = useContext(AuthContext)
  return (
    <section className="w-full mb-20  flex h-full justify-center items-center">
      <div className="flex flex-col  items-center justify-center w-full h-full px-6 py-8 mx-auto  :h-fit lg:py-0">
        
          <div className="w-full bg-gray-50 rounded-lg shadow   md:mt-0 sm:max-w-md xl:p-0 ">
              <div className="p-6 space-y-4 md:space-y-6 sm:p-8">
                  <h1 className="text-xl font-bold leading-tight tracking-tight text-gray-900 md:text-2xl ">
                      Login
                  </h1>
                  <form className="space-y-4 md:space-y-6" onSubmit={loginUser}>
                      <div>
                          <label htmlFor="email" className="block mb-2 text-sm font-medium text-gray-900 ">Your email</label>
                          <input type="email"     name="email" id="email" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5     " placeholder="name@company.com" required={true}/>
                      </div>
                      <div>
                          <label htmlFor="password" className="block mb-2 text-sm font-medium text-gray-900 ">Password</label>
                          <input type="password"  name="password" id="password" placeholder="••••••••" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5     " required={true}/>
                      </div>
                     
                      <button type="submit" className="w-full text-white  hover:bg-primary-700 focus:ring-4 focus:outline-none focus:ring-primary-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center bg-blue-700">Login</button>
                      {/* <p className="text-sm font-light text-gray-500 ">
                           create account <Link className="font-medium text-primary-600 hover:underline " to={"/register"}>Register</Link>
                      </p> */}
                  </form>
              </div>
          </div>
      </div>
    </section>


  )
}

export default Login