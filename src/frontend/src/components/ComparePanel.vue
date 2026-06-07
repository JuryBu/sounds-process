<template>
  <section class="compare-panel content-grid">
    <section class="upload-dock glass-panel">
      <div class="upload-copy">
        <span>模型对比</span>
        <h2>同一段音频跑多个模型，横向看 Top-1 与 Top-5</h2>
        <p>默认接入当前可用的真实 sklearn 基线；对比结果全部来自本地 joblib 模型推理。</p>
      </div>
      <div class="compare-controls">
        <label v-for="model in models" :key="model.id" class="model-check">
          <input v-model="selected" type="checkbox" :value="model.id" />
          <span>{{ model.name }}</span>
        </label>
        <label class="drop-zone compact" :class="{ hasFile: fileName }">
          <input type="file" accept="audio/*" :disabled="!models.length || busy" @change="handleFile" />
          <strong>{{ busy ? `对比中 ${fileName || '音频'}…` : fileName || '选择对比音频' }}</strong>
          <span>{{ errorMessage || compareMeta || (models.length ? '至少选择一个模型' : '后端真实模型未就绪') }}</span>
        </label>
      </div>
    </section>

    <section v-if="compareResult" class="glass-panel compare-results">
      <div class="panel-heading">
        <div>
          <span class="panel-kicker">Compare</span>
          <h2>{{ compareResult.filename }}</h2>
        </div>
      </div>
      <div class="compare-columns">
        <article v-for="row in compareResult.results" :key="row.model.id" class="compare-card">
          <span>{{ row.model.name }}</span>
          <h3>{{ row.winner?.common_name || '无结果' }}</h3>
          <p>{{ row.winner?.scientific_name }}</p>
          <strong>{{ confidencePercent(row.winner?.confidence) }}%</strong>
          <ol>
            <li v-for="item in row.top5" :key="item.species_id">
              {{ item.common_name }} <em>{{ confidencePercent(item.confidence) }}%</em>
            </li>
          </ol>
        </article>
      </div>
    </section>
  </section>
</template>

<script setup>
import { ref, watch } from 'vue'
import { compareAudio } from '../api'

const props = defineProps({
  models: {
    type: Array,
    required: true,
  },
})

const selected = ref([])
const busy = ref(false)
const fileName = ref('')
const compareMeta = ref('')
const errorMessage = ref('')
const compareResult = ref(null)

watch(
  () => props.models,
  (value) => {
    if (!selected.value.length) selected.value = value.slice(0, 3).map((item) => item.id)
  },
  { immediate: true },
)

async function handleFile(event) {
  const file = event.target.files?.[0]
  if (!file) return
  if (!selected.value.length) {
    errorMessage.value = '请至少选择一个模型'
    return
  }
  busy.value = true
  fileName.value = file.name
  compareMeta.value = `${(file.size / 1024 / 1024).toFixed(2)} MB · 已提交 ${selected.value.length} 个模型`
  errorMessage.value = ''
  try {
    compareResult.value = await compareAudio(file, selected.value)
  } catch (error) {
    errorMessage.value = error.message || '模型对比失败'
  } finally {
    busy.value = false
  }
}

function confidencePercent(value) {
  const numeric = Number(value || 0)
  return Math.round(numeric <= 1 ? numeric * 100 : numeric)
}
</script>
