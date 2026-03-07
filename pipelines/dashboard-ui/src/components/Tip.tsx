import { useState, useRef, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'
import s from './shared.module.css'

export function Tip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const iconRef = useRef<HTMLSpanElement>(null)
  const tipRef = useRef<HTMLSpanElement>(null)

  const reposition = useCallback(() => {
    const icon = iconRef.current
    if (!icon) return
    const rect = icon.getBoundingClientRect()
    setPos({
      top: rect.top - 6,
      left: rect.left + rect.width / 2,
    })
  }, [])

  // Adjust if tooltip overflows viewport edges
  useEffect(() => {
    if (!show || !tipRef.current || !pos) return
    const tip = tipRef.current
    const tipRect = tip.getBoundingClientRect()

    let adjustedLeft = pos.left
    const pad = 8
    if (tipRect.left < pad) {
      adjustedLeft = pad + tipRect.width / 2
    } else if (tipRect.right > window.innerWidth - pad) {
      adjustedLeft = window.innerWidth - pad - tipRect.width / 2
    }

    if (adjustedLeft !== pos.left) {
      setPos(prev => prev ? { ...prev, left: adjustedLeft } : prev)
    }
  }, [show, pos])

  return (
    <span
      className={s.tip}
      onMouseEnter={() => { reposition(); setShow(true) }}
      onMouseLeave={() => setShow(false)}
    >
      <span className={s.tipIcon} ref={iconRef}>?</span>
      {show && pos && createPortal(
        <span
          ref={tipRef}
          className={s.tipPortal}
          style={{ top: pos.top, left: pos.left }}
        >
          {text}
        </span>,
        document.body,
      )}
    </span>
  )
}
