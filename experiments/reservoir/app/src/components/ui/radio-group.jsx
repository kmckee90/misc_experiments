import * as React from "react"

const RadioGroup = ({ value, onValueChange, children, className, ...props }) => {
  return (
    <div className={`grid gap-2 ${className || ''}`} role="radiogroup" {...props}>
      {React.Children.map(children, (child, index) => {
        if (!React.isValidElement(child)) return child;
        return React.cloneElement(child, {
          key: index,
          name: props.name || 'radio-group',
          checked: child.props.value === value,
          onChange: () => onValueChange(child.props.value),
        })
      })}
    </div>
  )
}

const RadioGroupItem = React.forwardRef(({ className, value, ...props }, ref) => (
  <input
    type="radio"
    value={value}
    className={`aspect-square h-4 w-4 rounded-full border border-gray-300 text-blue-600 focus:ring-blue-500 ${className || ''}`}
    ref={ref}
    {...props}
  />
))
RadioGroupItem.displayName = "RadioGroupItem"

export { RadioGroup, RadioGroupItem }