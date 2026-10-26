import { useEffect, useMemo, useRef, useState } from 'react'
import {
  askCoach,
  createAnalysis,
  getAnalysis,
  getAnalysisStatus,
  getFrameUrl,
  selectSwimmer,
  startAnalysis,
} from './api'
import './App.css'

const STROKES = ['Freestyle', 'Backstroke', 'Breaststroke', 'Butterfly', 'Individual Medley']
const DISTANCES = {
  SCY: { default: [25, 50, 100, 200], 'Individual Medley': [100, 200] },
  SCM: { default: [25, 50, 100, 200], 'Individual Medley': [100, 200] },
  LCM: { default: [50, 100, 200], 'Individual Medley': [200] },
}
const POOLS = [
  { value: 'SCY', label: '25-yard pool' },
  { value: 'SCM', label: '25-meter pool' },
  { value: 'LCM', label: '50-meter pool' },
]

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" /></svg>
)

const WaveMark = () => (
  <span className="wave-mark" aria-hidden="true">
    <svg viewBox="0 0 36 36"><path d="M4 19c5.6 0 5.6-6 11.2-6s5.6 6 11.2 6c2.4 0 4-.9 5.6-2.2M4 25c5.6 0 5.6-6 11.2-6s5.6 6 11.2 6c2.4 0 4-.9 5.6-2.2" /></svg>
  </span>
)

function Header({ onNew, compact = false }) {
  return (
    <header className={`app-header ${compact ? 'compact' : ''}`}>
      <button className="brand" type="button" onClick={onNew} aria-label="SwimVision home">
        <WaveMark />
        <span>Swim<span>Vision</span></span>
      </button>
      <div className="header-note">Private video analysis</div>
      {compact && <button className="ghost-button" type="button" onClick={onNew}>Analyze another swim</button>}
    </header>
  )
}

function UploadScreen({ onCreated }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [stroke, setStroke] = useState('Freestyle')
  const [distance, setDistance] = useState(50)
  const [pool, setPool] = useState('SCY')
  const [officialTime, setOfficialTime] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const allowedDistances = DISTANCES[pool][stroke] || DISTANCES[pool].default
  const selectedDistance = allowedDistances.includes(distance) ? distance : allowedDistances[0]

  const chooseFile = (candidate) => {
    if (!candidate) return
    const extension = candidate.name.split('.').pop()?.toLowerCase()
    if (!candidate.type.startsWith('video/') && !['mkv', 'm4v'].includes(extension)) {
      setError('Choose a video file such as MP4, MOV, AVI, or MKV.')
      return
    }
    setError('')
    setFile(candidate)
  }

  const submit = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const created = await createAnalysis({ file, stroke, distance: selectedDistance, pool, officialTime })
      onCreated(created)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page-shell upload-page">
      <Header onNew={() => window.location.reload()} />
      <main className="upload-layout">
        <section className="hero-copy">
          <div className="eyebrow"><span /> Your race, explained</div>
          <h1>See more than<br /><em>the finish time.</em></h1>
          <p>Upload a race video. SwimVision follows your swim and turns it into clear, stroke-specific coaching you can use at your next practice.</p>
          <div className="promise-row">
            <div><strong>One tap</strong><span>to choose yourself</span></div>
            <div><strong>Whole race</strong><span>analyzed start to finish</span></div>
            <div><strong>Plain English</strong><span>without technical clutter</span></div>
          </div>
        </section>

        <section className="upload-card" aria-label="Upload a swim video">
          <div
            className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]) }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="video/*,.mkv,.m4v" hidden onChange={(event) => chooseFile(event.target.files[0])} />
            <span className="upload-icon"><UploadIcon /></span>
            {file ? (
              <><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(1)} MB · Click to replace</span></>
            ) : (
              <><strong>Drop your race video here</strong><span>or click to choose a file</span><small>MP4, MOV, AVI, M4V or MKV</small></>
            )}
          </div>

          <div className="race-form">
            <label>Stroke<select value={stroke} onChange={(event) => setStroke(event.target.value)}>{STROKES.map((item) => <option key={item}>{item}</option>)}</select></label>
            <label>Distance<select value={selectedDistance} onChange={(event) => setDistance(Number(event.target.value))}>{allowedDistances.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label>Pool<select value={pool} onChange={(event) => setPool(event.target.value)}>{POOLS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label className="time-field">Official time <span>optional</span><input value={officialTime} onChange={(event) => setOfficialTime(event.target.value)} placeholder="54.32 or 1:54.32" /></label>
          </div>
          <p className="time-note">Your official time improves the report. It never changes the video or tracking.</p>
          {error && <p className="form-error" role="alert">{error}</p>}
          <button className="primary-button" type="button" disabled={!file || busy} onClick={submit}>
            {busy ? 'Preparing your video…' : 'Continue'} <span aria-hidden="true">→</span>
          </button>
        </section>
      </main>
    </div>
  )
}

function SelectScreen({ analysis, onConfirmed, onBack }) {
  const [point, setPoint] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const pick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    setPoint({ x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height })
    setError('')
  }

  const confirm = async () => {
    if (!point) return
    setBusy(true)
    try {
      await selectSwimmer(analysis.analysis_id, point)
      await startAnalysis(analysis.analysis_id, analysis.official_time_s)
      onConfirmed()
    } catch (err) {
      setError(err.message)
      setBusy(false)
    }
  }

  return (
    <div className="page-shell">
      <Header onNew={onBack} compact />
      <main className="focus-layout">
        <div className="step-pill">One quick step</div>
        <h1>Which swimmer are you?</h1>
        <p>Tap directly on yourself in the opening frame. That’s the only selection you’ll make.</p>
        <div className="selection-frame" onClick={pick} role="button" tabIndex={0}>
          <img src={getFrameUrl(analysis.analysis_id)} alt="Opening frame of the uploaded race" />
          {point && <span className="selection-point" style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}><i /></span>}
        </div>
        <div className="selection-actions">
          <button className="text-button" type="button" onClick={onBack}>← Back</button>
          <button className="primary-button compact-button" type="button" disabled={!point || busy} onClick={confirm}>{busy ? 'Starting…' : 'That’s me'} <span>→</span></button>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
      </main>
    </div>
  )
}

function ProcessingScreen({ analysisId, onComplete, onFail, onNew }) {
  const [status, setStatus] = useState({ stage: 'Preparing your race', progress: 0.03 })
  useEffect(() => {
    let active = true
    const check = async () => {
      try {
        const next = await getAnalysisStatus(analysisId)
        if (!active) return
        setStatus(next)
        if (next.state === 'complete') return onComplete()
        if (next.state === 'failed') return onFail(next.message || 'Analysis could not be completed.')
        window.setTimeout(check, 1200)
      } catch (err) {
        if (active) onFail(err.message)
      }
    }
    check()
    return () => { active = false }
  }, [analysisId, onComplete, onFail])

  const steps = ['Preparing your race', 'Following your swim', 'Reading your race', 'Building your coaching plan']
  const activeIndex = Math.max(0, steps.indexOf(status.stage))
  return (
    <div className="page-shell processing-page">
      <Header onNew={onNew} compact />
      <main className="processing-card">
        <div className="water-orbit"><WaveMark /><span /><span /></div>
        <p className="eyebrow centered">SwimVision is working</p>
        <h1>{status.stage}</h1>
        <p>This can take a few minutes for a full race. You can leave this tab open.</p>
        <div className="progress-track"><span style={{ width: `${Math.max(4, Math.round((status.progress || 0) * 100))}%` }} /></div>
        <div className="friendly-steps">
          {steps.map((step, index) => <div key={step} className={index < activeIndex ? 'done' : index === activeIndex ? 'active' : ''}><span>{index < activeIndex ? '✓' : index + 1}</span>{step}</div>)}
        </div>
      </main>
    </div>
  )
}

function PointCard({ point, kind }) {
  return (
    <article className={`point-card ${kind}`}>
      <span className="point-icon">{kind === 'strength' ? '✓' : '↗'}</span>
      <div><h3>{point.title}</h3><p>{point.explanation}</p>{point.action && <div className="action-line"><strong>Try this</strong>{point.action}</div>}</div>
    </article>
  )
}

function OverviewTab({ report }) {
  return <div className="results-grid">
    <section className="result-section wide"><div className="section-heading"><span>01</span><div><p>Your clearest takeaway</p><h2>{report.headline}</h2></div></div><p className="lead-copy">{report.overview}</p></section>
    <section className="result-section"><div className="section-kicker">What looked good</div>{report.strengths?.length ? report.strengths.map((point) => <PointCard key={point.title} point={point} kind="strength" />) : <p className="empty-note">This angle did not show a clear repeatable strength. That does not mean your technique was poor.</p>}</section>
    <section className="result-section"><div className="section-kicker">Work on this first</div>{report.priorities?.length ? report.priorities.map((point) => <PointCard key={point.title} point={point} kind="priority" />) : <p className="empty-note">No clear technique problem was measurable in this video.</p>}</section>
    {report.next_practice && <section className="practice-banner wide"><div><span>Next practice</span><h2>{report.next_practice.name}</h2><p>{report.next_practice.purpose}</p></div><div className="practice-set"><strong>{report.next_practice.set_text}</strong><span>Success looks like: {report.next_practice.success_check}</span></div></section>}
  </div>
}

function TechniqueTab({ report }) {
  const technique = report.technique || {}
  return <div className="tab-stack">
    <section className="technique-intro"><div><span className="stroke-label">{technique.stroke || report.race_label}</span><h2>{technique.title || 'Technique breakdown'}</h2><p>{technique.summary || report.overview}</p></div><div className="view-score"><strong>{technique.visible_cycles ?? '—'}</strong><span>clear cycles reviewed</span></div></section>
    <div className="technique-list">{(technique.sections || []).map((section, index) => <article key={section.title}><span>{String(index + 1).padStart(2, '0')}</span><div><h3>{section.title}</h3><p>{section.explanation}</p><strong>{section.action}</strong></div><em className={section.rating}>{section.rating_label}</em></article>)}</div>
    {technique.camera_note && <div className="camera-note"><span>Camera note</span><p>{technique.camera_note}</p></div>}
  </div>
}

function RaceTab({ report }) {
  return <div className="tab-stack"><section className="race-table"><div className="race-row header"><span>Length</span><span>Split</span><span>Pace</span><span>Cycles</span></div>{(report.lengths || []).map((length) => <div className="race-row" key={length.number}><strong>Length {length.number}</strong><span>{length.split_display}</span><span>{length.pace_note || '—'}</span><span>{length.stroke_cycles ?? '—'}</span></div>)}</section>{!report.lengths?.length && <div className="empty-panel"><h2>No trustworthy splits yet</h2><p>The race did not show every wall clearly enough, so SwimVision did not invent missing split times.</p></div>}</div>
}

function PlanTab({ report }) {
  return <div className="tab-stack"><div className="plan-grid">{(report.training_sets || []).map((item, index) => <article key={item.name}><span>{String(index + 1).padStart(2, '0')}</span><h2>{item.name}</h2><p>{item.purpose}</p><div>{item.set_text}</div><small>Goal: {item.success_check}</small></article>)}</div>{report.nutrition_tips?.length > 0 && <section className="fuel-card"><div><span>Fuel & recovery</span><h2>Simple habits around practice</h2></div><ul>{report.nutrition_tips.map((tip) => <li key={tip}>{tip}</li>)}</ul><small>General sports guidance only. A parent, coach, doctor, or sports dietitian should personalize food and hydration needs.</small></section>}</div>
}

function CoachTab({ report, analysisId }) {
  const [messages, setMessages] = useState([{ role: 'coach', content: report.overview }])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const send = async (text) => {
    const cleaned = text.trim()
    if (!cleaned || busy) return
    setMessages((current) => [...current, { role: 'user', content: cleaned }])
    setQuestion('')
    setBusy(true)
    try {
      const response = await askCoach(analysisId, cleaned)
      setMessages((current) => [...current, { role: 'coach', content: response.answer }])
    } catch (err) {
      setMessages((current) => [...current, { role: 'coach', content: err.message }])
    } finally { setBusy(false) }
  }
  return <div className="coach-layout"><aside><span>Ask about this race</span><h2>Your SwimVision coach</h2><p>Answers use only measurements from this swim. If the video cannot answer, your coach will say so.</p><div className="prompt-list">{report.suggested_questions?.map((prompt) => <button key={prompt} type="button" onClick={() => send(prompt)}>{prompt}<span>→</span></button>)}</div></aside><section className="chat-panel"><div className="messages">{messages.map((message, index) => <div key={`${message.role}-${index}`} className={`message ${message.role}`}>{message.role === 'coach' && <WaveMark />}<p>{message.content}</p></div>)}{busy && <div className="message coach typing"><WaveMark /><p><i /><i /><i /></p></div>}</div><form onSubmit={(event) => { event.preventDefault(); send(question) }}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about your technique, pacing, or next practice…" /><button type="submit" disabled={busy || !question.trim()}>Send</button></form></section></div>
}

function ResultsScreen({ report, analysisId, onNew }) {
  const tabs = ['Overview', 'Technique', 'Race', 'Practice plan', 'Ask coach']
  const [tab, setTab] = useState('Overview')
  return <div className="page-shell results-page"><Header onNew={onNew} compact /><main className="results-shell"><section className="result-hero"><div><div className="eyebrow"><span /> Analysis complete</div><p>{report.race_label}</p><h1>{report.headline}</h1></div><div className="time-card"><span>{report.time_label}</span><strong>{report.race_time_display}</strong>{report.time_label.startsWith('Official') && <small>Video estimate {report.video_estimate_display}</small>}</div></section><nav className="result-tabs" aria-label="Analysis sections">{tabs.map((item) => <button type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)} key={item}>{item}</button>)}</nav><div className="tab-content">{tab === 'Overview' && <OverviewTab report={report} />}{tab === 'Technique' && <TechniqueTab report={report} />}{tab === 'Race' && <RaceTab report={report} />}{tab === 'Practice plan' && <PlanTab report={report} />}{tab === 'Ask coach' && <CoachTab report={report} analysisId={analysisId} />}</div></main></div>
}

function App() {
  const [screen, setScreen] = useState('upload')
  const [analysis, setAnalysis] = useState(null)
  const [report, setReport] = useState(null)
  const [fatalError, setFatalError] = useState('')

  const reset = () => { setScreen('upload'); setAnalysis(null); setReport(null); setFatalError('') }
  const complete = useMemo(() => async () => {
    try { setReport(await getAnalysis(analysis.analysis_id)); setScreen('results') }
    catch (err) { setFatalError(err.message); setScreen('error') }
  }, [analysis])

  if (screen === 'select') return <SelectScreen analysis={analysis} onBack={reset} onConfirmed={() => setScreen('processing')} />
  if (screen === 'processing') return <ProcessingScreen analysisId={analysis.analysis_id} onComplete={complete} onFail={(message) => { setFatalError(message); setScreen('error') }} onNew={reset} />
  if (screen === 'results') return <ResultsScreen report={report} analysisId={analysis.analysis_id} onNew={reset} />
  if (screen === 'error') return <div className="page-shell"><Header onNew={reset} compact /><main className="error-page"><span>We couldn’t finish this analysis</span><h1>Let’s try that again.</h1><p>{fatalError}</p><button className="primary-button compact-button" type="button" onClick={reset}>Start over</button></main></div>
  return <UploadScreen onCreated={(created) => { setAnalysis(created); setScreen('select') }} />
}

export default App
