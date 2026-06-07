<template>
  <section class="glass-panel audio-panel waveform-panel">
    <div class="panel-heading">
      <div>
        <span class="panel-kicker">Waveform</span>
        <h2>波形图</h2>
      </div>
      <div v-if="hasData" class="audio-actions">
        <span>{{ durationLabel }}</span>
      </div>
    </div>
    <div v-if="hasData" class="waveform-canvas-wrap">
      <canvas ref="canvasRef" width="1200" height="220" aria-label="鸟鸣音频波形"></canvas>
      <div class="timeline">
        <span v-for="tick in ticks" :key="tick">{{ tick }}</span>
      </div>
    </div>
    <div v-else class="empty-state">
      <span>上传或录制音频后显示波形</span>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps({
  waveform: {
    type: Object,
    default: null,
  },
  audio: {
    type: Object,
    default: null,
  },
})

const canvasRef = ref(null)

const hasData = computed(() => props.waveform?.points?.length > 0)

const durationSeconds = computed(() => {
  if (props.audio?.duration_seconds) return props.audio.duration_seconds
  if (props.audio?.sample_rate && props.waveform?.sample_count) {
    return props.waveform.sample_count / props.audio.sample_rate
  }
  if (props.waveform?.points?.length > 0) return 5
  return 0
})

const durationLabel = computed(() => {
  const s = Math.round(durationSeconds.value)
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
})

const ticks = computed(() => {
  const dur = durationSeconds.value || 5
  const step = dur <= 10 ? 1 : dur <= 30 ? 5 : 10
  const result = []
  for (let t = 0; t <= dur; t += step) {
    const m = Math.floor(t / 60)
    const s = t % 60
    result.push(`${m}:${String(s).padStart(2, '0')}`)
  }
  return result
})

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const points = props.waveform?.points
  if (!points?.length) return
  if (canvas.offsetWidth === 0) {
    setTimeout(draw, 100)
    return
  }

  const ctx = canvas.getContext('2d')
  const width = canvas.width
  const height = canvas.height
  const center = height / 2
  const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#4ade80'

  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.018)'
  for (let x = 0; x < width; x += 96) ctx.fillRect(x, 0, 1, height)
  for (let y = 34; y < height; y += 48) ctx.fillRect(0, y, width, 1)

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)'
  ctx.beginPath()
  ctx.moveTo(0, center)
  ctx.lineTo(width, center)
  ctx.stroke()

  const gradient = ctx.createLinearGradient(0, 20, 0, height - 20)
  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.78)')
  gradient.addColorStop(0.18, primary)
  gradient.addColorStop(0.5, primary)
  gradient.addColorStop(0.82, primary)
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.66)')
  ctx.strokeStyle = gradient
  ctx.lineWidth = 1.4
  ctx.shadowColor = primary
  ctx.shadowBlur = 12

  points.forEach((value, index) => {
    const x = (index / Math.max(1, points.length - 1)) * width
    const amp = Math.min(1, Math.abs(value)) * 92
    ctx.beginPath()
    ctx.moveTo(x, center - amp)
    ctx.lineTo(x, center + amp)
    ctx.stroke()
  })
}

onMounted(() => nextTick(draw))
watch(() => props.waveform, () => nextTick(draw), { deep: true })
</script>
