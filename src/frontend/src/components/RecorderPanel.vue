<template>
  <section class="upload-dock glass-panel recorder-panel">
    <div class="upload-copy">
      <span>麦克风录音</span>
      <h2>现场采一段鸟鸣或人声，直接送到后端识别</h2>
      <p>使用 Web Audio API 绘制实时波形；停止后会把录音文件提交给当前模型。</p>
    </div>

    <div class="recorder-console">
      <canvas ref="canvasRef" width="520" height="150" aria-label="实时录音波形"></canvas>
      <div class="recorder-actions">
        <button class="btn-primary" type="button" :disabled="disabled || busy || recording" @click="startRecording">开始录音</button>
        <button class="btn-ghost" type="button" :disabled="!recording" @click="stopRecording">停止并识别</button>
      </div>
      <p class="inline-status">{{ statusText }}</p>
      <p v-if="error || localError" class="inline-error">{{ error || localError }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'

defineProps({
  busy: {
    type: Boolean,
    default: false,
  },
  error: {
    type: String,
    default: '',
  },
  disabled: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['recorded'])
const canvasRef = ref(null)
const recording = ref(false)
const localError = ref('')
const seconds = ref(0)
let mediaRecorder = null
let mediaStream = null
let audioContext = null
let analyser = null
let animationId = 0
let timerId = 0
let chunks = []

const statusText = computed(() => {
  if (recording.value) return `正在录音 ${seconds.value}s · 波形实时刷新中`
  return '浏览器会请求麦克风权限；录音仅在本地送到后端，不会额外上传到第三方。'
})

let scriptProcessorNode = null
let recordedSamples = []

async function startRecording() {
  localError.value = ''
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 32000, channelCount: 1 } })
    audioContext = new AudioContext({ sampleRate: 32000 })
    const source = audioContext.createMediaStreamSource(mediaStream)
    analyser = audioContext.createAnalyser()
    analyser.fftSize = 2048
    source.connect(analyser)

    recordedSamples = []
    scriptProcessorNode = audioContext.createScriptProcessor(4096, 1, 1)
    scriptProcessorNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0)
      recordedSamples.push(new Float32Array(input))
    }
    source.connect(scriptProcessorNode)
    scriptProcessorNode.connect(audioContext.destination)

    recording.value = true
    seconds.value = 0
    timerId = window.setInterval(() => {
      seconds.value += 1
    }, 1000)
    drawLiveWaveform()
  } catch (error) {
    localError.value = `无法打开麦克风：${error.message || error}`
    cleanup()
  }
}

function stopRecording() {
  recording.value = false
  window.clearInterval(timerId)
  window.cancelAnimationFrame(animationId)
  submitRecording()
}

function submitRecording() {
  const totalLength = recordedSamples.reduce((acc, buf) => acc + buf.length, 0)
  const merged = new Float32Array(totalLength)
  let offset = 0
  for (const buf of recordedSamples) {
    merged.set(buf, offset)
    offset += buf.length
  }

  const sampleRate = audioContext?.sampleRate || 32000
  const wavBlob = encodeWav(merged, sampleRate)
  const file = new File([wavBlob], `birdvoice-recording-${Date.now()}.wav`, { type: 'audio/wav' })
  emit('recorded', file)
  cleanup()
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

function drawLiveWaveform() {
  const canvas = canvasRef.value
  if (!canvas || !analyser) return
  const ctx = canvas.getContext('2d')
  const data = new Uint8Array(analyser.fftSize)
  const width = canvas.width
  const height = canvas.height
  const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#4ade80'

  analyser.getByteTimeDomainData(data)
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.22)'
  ctx.fillRect(0, 0, width, height)
  ctx.strokeStyle = primary
  ctx.lineWidth = 2
  ctx.shadowColor = primary
  ctx.shadowBlur = 12
  ctx.beginPath()
  data.forEach((value, index) => {
    const x = (index / (data.length - 1)) * width
    const y = (value / 255) * height
    if (index === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
  animationId = window.requestAnimationFrame(drawLiveWaveform)
}

function cleanup() {
  window.clearInterval(timerId)
  window.cancelAnimationFrame(animationId)
  scriptProcessorNode?.disconnect()
  scriptProcessorNode = null
  mediaStream?.getTracks().forEach((track) => track.stop())
  audioContext?.close?.()
  mediaStream = null
  audioContext = null
  analyser = null
  mediaRecorder = null
  recordedSamples = []
}

onBeforeUnmount(cleanup)
</script>
