import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './App.css'
import Login from './pages/Login/Login'
import { AuthProvider } from './context/AuthContext'
import HomePage from './pages/HomePage/HomePage'

function App() {
  const router = createBrowserRouter([
    {
      path: '/',
      element: <AuthProvider />,
      children: [
        {
          path: '/login',
          element: <Login />
        },
        {
          path: '/',
          element: <HomePage />
        },
      ]

    }
  ])

  return (
   
    <RouterProvider router={router} />
    
  )
}

export default App
