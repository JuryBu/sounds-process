<template>
  <BackgroundLayer />

  <main class="app-shell">
    <SideBar v-model:active="activeNav" />

    <section class="workspace glass-panel">
      <TopBar v-model:model="selectedModel" :models="models" :backend-ready="backendReady" />

      <div v-show="activeNav === 'upload'" class="content-grid">
        <UploadDock :busy="predicting" :disabled="!backendReady" :error="errorMessage" @file-selected="handleFileSelected" />
        <WaveformPanel :waveform="prediction?.waveform" :audio="prediction?.audio" />
        <MelSpectrogramPanel :mel="prediction?.mel_spectrogram" />
        <ResultCards :items="prediction?.top5" />
      </div>
      <div v-show="activeNav === 'record'" class="content-grid">
        <RecorderPanel :busy="predicting" :disabled="!backendReady" :error="errorMessage" @recorded="handleFileSelected" />
        <WaveformPanel :waveform="prediction?.waveform" :audio="prediction?.audio" />
        <MelSpectrogramPanel :mel="prediction?.mel_spectrogram" />
        <ResultCards :items="prediction?.top5" />
      </div>
      <ComparePanel v-if="activeNav === 'compare'" :models="modelRows" />
      <HistoryPanel v-if="activeNav === 'history'" :items="historyRows" @clear="clearHistory" />
    </section>
  </main>

  <button class="floating-settings" type="button" aria-label="打开设置" @click="settingsOpen = true">⚙</button>
  <SettingsModal v-if="settingsOpen || activeNav === 'settings'" @close="closeSettings" />
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { fetchModels, predictAudio } from './api'
import BackgroundLayer from './components/BackgroundLayer.vue'
import MelSpectrogramPanel from './components/MelSpectrogramPanel.vue'
import ComparePanel from './components/ComparePanel.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import RecorderPanel from './components/RecorderPanel.vue'
import ResultCards from './components/ResultCards.vue'
import SettingsModal from './components/SettingsModal.vue'
import SideBar from './components/SideBar.vue'
import TopBar from './components/TopBar.vue'
import UploadDock from './components/UploadDock.vue'
import WaveformPanel from './components/WaveformPanel.vue'

const activeNav = ref('upload')
const settingsOpen = ref(false)
const selectedModel = ref('knn-real')
const modelRows = ref([])
const prediction = ref(null)
const predicting = ref(false)
const errorMessage = ref('')
const historyRows = ref([])
const lastFile = ref(null)

const models = computed(() => modelRows.value.map((item) => ({ value: item.id, label: item.name })))
const backendReady = computed(() => modelRows.value.length > 0 && Boolean(selectedModel.value))

watch(activeNav, (value) => {
  if (value === 'settings') settingsOpen.value = true
})

function closeSettings() {
  settingsOpen.value = false
  if (activeNav.value === 'settings') activeNav.value = 'upload'
}

async function handleFileSelected(file) {
  lastFile.value = file
  await runPrediction(file)
}

async function runPrediction(file) {
  if (!file) return
  if (!backendReady.value) {
    errorMessage.value = '后端或真实模型未就绪，请先启动后端并确认模型文件存在'
    return
  }
  predicting.value = true
  errorMessage.value = ''
  prediction.value = null
  try {
    prediction.value = await predictAudio(file, selectedModel.value)
    addHistory(file, prediction.value)
  } catch (error) {
    prediction.value = null
    errorMessage.value = error.message || '识别失败'
  } finally {
    predicting.value = false
  }
}

watch(selectedModel, () => {
  if (lastFile.value && backendReady.value) {
    runPrediction(lastFile.value)
  }
})

onMounted(async () => {
  historyRows.value = loadHistory()
  try {
    const data = await fetchModels()
    modelRows.value = (data.models || []).filter(m => m.kind !== 'mock')
    selectedModel.value = data.default_model || modelRows.value[0]?.id || ''
  } catch {
    modelRows.value = []
    selectedModel.value = ''
    errorMessage.value = '无法连接后端服务，请确认后端已在 8017 端口启动'
  }
})

function addHistory(file, result) {
  const top = result?.top5?.[0]
  const row = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    time: new Date().toLocaleString('zh-CN'),
    filename: file?.name || result?.filename || 'recording.webm',
    model: result?.model?.name || selectedModel.value,
    winner: top ? `${top.common_name} / ${top.scientific_name}` : '未识别',
    confidence: top?.confidence ?? 0,
    note: result?.note || '',
  }
  historyRows.value = [row, ...historyRows.value].slice(0, 30)
  localStorage.setItem('birdvoice-history', JSON.stringify(historyRows.value))
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem('birdvoice-history') || '[]')
  } catch {
    return []
  }
}

function clearHistory() {
  historyRows.value = []
  localStorage.removeItem('birdvoice-history')
}
</script>
