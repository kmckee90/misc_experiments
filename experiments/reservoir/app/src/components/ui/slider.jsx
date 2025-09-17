import * as React from "react"

const Slider = React.forwardRef(({ className, min = 0, max = 100, step = 1, value, onValueChange, ...props }, ref) => {
  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value)
    onValueChange([newValue])
  }

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      onChange={handleChange}
      className={`w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider ${className || ''}`}
      ref={ref}
      {...props}
    />
  )
})
Slider.displayName = "Slider"

export { Slider }