import React from 'react';

const Button = ({ children, onClick, variant = 'primary' }) => {
  const buttonStyles =
    variant === 'primary'
      ? 'bg-[#1980e6] text-white'
      : 'bg-[#f0f2f4] text-[#111418]';
  return (
    <button
      className={`flex items-center justify-center rounded-xl h-10 px-4 text-sm font-bold ${buttonStyles}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};

export default Button;
