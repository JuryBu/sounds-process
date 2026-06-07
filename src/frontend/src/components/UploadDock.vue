<template>
  <section class="upload-dock glass-panel">
    <div class="upload-copy">
      <span>上传音频</span>
      <h2>拖入一段森林录音，预览识别流程</h2>
      <p>选择 `.wav`、`.mp3`、`.ogg` 后会调用 FastAPI 后端，刷新梅尔频谱、波形和 Top-5 结果。</p>
    </div>
    <label class="drop-zone" :class="{ hasFile: fileName }">
      <input type="file" accept="audio/*" :disabled="disabled || busy" @change="handleFile" />
      <strong>{{ busy ? `正在识别 ${fileName || '音频'}…` : fileName || '选择音频文件' }}</strong>
      <span>{{ error || fileMeta || (disabled ? '后端真实模型未就绪' : '支持 WAV / MP3 / OGG') }}</span>
    </label>
  </section>
</template>

<script setup>
import { ref } from 'vue'

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

const emit = defineEmits(['file-selected'])
const fileName = ref('')
const fileMeta = ref('')

function handleFile(event) {
  const file = event.target.files?.[0]
  if (!file) return
  fileName.value = file.name
  fileMeta.value = `${(file.size / 1024 / 1024).toFixed(2)} MB · 已提交后端`
  emit('file-selected', file)
}
</script>
