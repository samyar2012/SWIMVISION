import { useCallback, useEffect, useRef, useState } from 'react'
import {
  askCoach,
  createAnalysis,
  fetchLiveFrame,
  getAnalysis,
  getAnalysisStatus,
  getFrameUrl,
  getVideoUrl,
  selectSwimmer,
  startAnalysis,
} from './api'
import './App.css'

const STROKES = ['Freestyle', 'Backstroke', 'Breaststroke', 'Butterfly']
const DISTANCES = { SCY: [25, 50, 100, 200], SCM: [25, 50, 100, 200], LCM: [50, 100, 200] }
const POOLS = [
  { value: 'SCY', label: '25-yard pool' },
  { value: 'SCM', label: '25-meter pool' },
  { value: 'LCM', label: '50-meter pool' },
]
const STAGES = ['Preparing your race', 'Following your swim', 'Timing your race', 'Writing your report']

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
  const allowedDistances = DISTANCES[pool]
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
          <p>Upload a race video. SwimVision follows you through the whole race, times every length, and tells you in plain words what went well and what to work on.</p>
          <div className="promise-row">
            <div><strong>One tap</strong><span>to choose yourself</span></div>
            <div><strong>Real splits</strong><span>start, turns and finish</span></div>
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
          <p className="time-note">If you know your official time, add it. It makes the total and the last split exact.</p>
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
  const [box, setBox] = useState(null)
  const [finding, setFinding] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  const pick = async (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const next = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height }
    if (next.x < 0 || next.x > 1 || next.y < 0 || next.y > 1) return
    setPoint(next)
    setBox(null)
    setError('')
    setFinding(true)
    try {
      const selected = await selectSwimmer(analysis.analysis_id, next)
      setBox(selected.box)
    } catch (err) {
      setError(err.message)
    } finally {
      setFinding(false)
    }
  }

  const confirm = async () => {
    if (!box) return
    setStarting(true)
    try {
      await startAnalysis(analysis.analysis_id, analysis.official_time_s)
      onConfirmed()
    } catch (err) {
      setError(err.message)
      setStarting(false)
    }
  }

  return (
    <div className="page-shell">
      <Header onNew={onBack} compact />
      <main className="focus-layout">
        <div className="step-pill">One quick step</div>
        <h1>Which swimmer are you?</h1>
        <p>{box ? 'Is the highlighted swimmer you? If not, tap yourself again.' : 'Tap directly on yourself in the opening frame.'}</p>
        <div className="selection-frame">
          <div className="selection-image" onClick={pick} role="button" tabIndex={0}>
            <img src={getFrameUrl(analysis.analysis_id)} alt="Opening frame of the uploaded race" draggable={false} />
            {box && (
              <span
                className="selection-box"
                style={{ left: `${box.x1 * 100}%`, top: `${box.y1 * 100}%`, width: `${(box.x2 - box.x1) * 100}%`, height: `${(box.y2 - box.y1) * 100}%` }}
              />
            )}
            {point && !box && <span className={`selection-point ${finding ? 'finding' : ''}`} style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}><i /></span>}
          </div>
        </div>
        <div className="selection-actions">
          <button className="text-button" type="button" onClick={onBack}>← Back</button>
          <button className="primary-button compact-button" type="button" disabled={!box || finding || starting} onClick={confirm}>
            {starting ? 'Starting…' : finding ? 'Finding you…' : 'That’s me'} <span>→</span>
          </button>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
      </main>
    </div>
  )
}

function LiveView({ analysisId, active }) {
  const [frame, setFrame] = useState(null)
  useEffect(() => {
    if (!active) return undefined
    let alive = true
    let timer
    const load = async () => {
      try {
        const url = await fetchLiveFrame(analysisId)
        if (!alive) { if (url) URL.revokeObjectURL(url); return }
        if (url) setFrame(url)
      } catch { /* the next poll retries */ }
      if (alive) timer = window.setTimeout(load, 180)
    }
    load()
    return () => { alive = false; window.clearTimeout(timer) }
  }, [analysisId, active])
  useEffect(() => () => { if (frame) URL.revokeObjectURL(frame) }, [frame])

  return (
    <div className={`live-view ${frame ? 'has-frame' : ''}`}>
      {frame ? <img src={frame} alt="Your race being tracked, with you highlighted" /> : <div className="live-placeholder"><WaveMark /><span>Loading your video…</span></div>}
      {frame && active && <span className="live-badge"><i />Live</span>}
    </div>
  )
}

function ProcessingScreen({ analysisId, onComplete, onFail, onNew }) {
  const [status, setStatus] = useState({ stage: 'Preparing your race', progress: 0.03 })
  useEffect(() => {
    let active = true
    let timer
    const check = async () => {
      try {
        const next = await getAnalysisStatus(analysisId)
        if (!active) return
        setStatus(next)
        if (next.state === 'complete') return onComplete()
        if (next.state === 'failed') return onFail(next.message || 'Analysis could not be completed.')
        timer = window.setTimeout(check, 1000)
      } catch (err) {
        if (active) onFail(err.message)
      }
    }
    check()
    return () => { active = false; window.clearTimeout(timer) }
  }, [analysisId, onComplete, onFail])

  const activeIndex = Math.max(0, STAGES.indexOf(status.stage))
  const tracking = activeIndex <= 1
  return (
    <div className="page-shell processing-page">
      <Header onNew={onNew} compact />
      <main className="processing-live">
        <LiveView analysisId={analysisId} active={tracking} />
        <section className="processing-side">
          <p className="eyebrow"><span /> SwimVision is working</p>
          <h1>{status.stage}</h1>
          <p>{tracking ? 'Watch SwimVision follow you through the race. You stay highlighted the whole way.' : 'Tracking is done. Now your start, turns and finish are being timed.'}</p>
          <div className="progress-track"><span style={{ width: `${Math.max(4, Math.round((status.progress || 0) * 100))}%` }} /></div>
          <div className="friendly-steps vertical">
            {STAGES.map((step, index) => <div key={step} className={index < activeIndex ? 'done' : index === activeIndex ? 'active' : ''}><span>{index < activeIndex ? '✓' : index + 1}</span>{step}</div>)}
          </div>
        </section>
      </main>
    </div>
  )
}

function RaceVideo({ report, videoRef }) {
  const seek = (seconds) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, seconds - 0.5)
    video.play().catch(() => {})
  }
  if (!report.has_video) return null
  return (
    <section className="race-video">
      <div className="video-kicker">Your race, with you highlighted</div>
      <video ref={videoRef} src={getVideoUrl(report.analysis_id)} controls playsInline preload="metadata" />
      {report.moments?.length > 0 && (
        <div className="moment-row" aria-label="Jump to a moment in the race">
          {report.moments.map((moment) => (
            <button type="button" key={`${moment.label}-${moment.video_s}`} onClick={() => seek(moment.video_s)}>
              <strong>{moment.label}</strong><span>{moment.race_clock}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  )
}

function OverviewTab({ report, onTab }) {
  return (
    <div className="tab-stack">
      <section className="result-section"><p className="lead-copy flush">{report.overview}</p></section>
      {report.highlights?.length > 0 && (
        <div className="stat-grid">
          {report.highlights.map((item) => <div className="stat-card" key={item.label}><span>{item.label}</span><strong>{item.value}</strong><small>{item.detail}</small></div>)}
        </div>
      )}
      {report.focus?.length > 0 && (
        <section className="result-section">
          <div className="section-kicker">The biggest thing to work on</div>
          <h2 className="section-title">{report.focus[0].title}</h2>
          <p className="body-copy">{report.focus[0].why}</p>
          <button className="text-link" type="button" onClick={() => onTab('What to work on')}>See what to work on →</button>
        </section>
      )}
      <Notes notes={report.notes} />
    </div>
  )
}

function Notes({ notes }) {
  if (!notes?.length) return null
  return <div className="camera-note"><span>Good to know</span>{notes.map((note) => <p key={note}>{note}</p>)}</div>
}

function TechniqueTab({ report }) {
  if (!report.technique?.length) {
    return <div className="empty-panel"><h2>Not enough of the race was visible</h2><p>SwimVision needs to see the start or both walls clearly to describe how the race went.</p></div>
  }
  return (
    <div className="tab-stack">
      <div className="technique-notes">
        {report.technique.map((note) => (
          <article key={note.topic} className={`technique-note ${note.tone}`}>
            <h3>{note.topic}</h3>
            <p className="observation">{note.observation}</p>
            <p>{note.meaning}</p>
          </article>
        ))}
      </div>
      <div className="camera-note"><span>What this is based on</span><p>Start, turns, finish and how fast your arms were turning over come from this video. Breathing, kick and underwater work are not guessed.</p></div>
    </div>
  )
}

function formatDelta(delta) {
  if (delta == null) return '—'
  const sign = delta > 0 ? '+' : ''
  return `${sign}${delta.toFixed(2)}`
}

function RaceTab({ report, videoRef }) {
  const splits = report.splits || []
  const slowest = Math.max(...splits.map((item) => item.split_s), 1)
  const seek = (seconds) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, seconds - 0.5)
    video.play().catch(() => {})
    video.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
  if (!splits.length) {
    return <div className="tab-stack"><div className="empty-panel"><h2>No splits for this video</h2><p>Not every wall was visible, so SwimVision did not make up split times.</p></div><Notes notes={report.notes} /></div>
  }
  return (
    <div className="tab-stack">
      <section className="race-table">
        <div className="race-row header"><span>Length</span><span>Split</span><span>Vs last</span><span>Total</span><span>Pace / 100</span><span>Stroke rate</span></div>
        {splits.map((length) => (
          <button type="button" className={`race-row ${length.number === report.fastest_length ? 'fastest' : ''}`} key={length.number} onClick={() => seek(length.video_start_s)} title="Watch this length">
            <strong>{length.label}<small>{length.distance_label}</small></strong>
            <span className="split-cell"><b>{length.split}</b><i style={{ width: `${(length.split_s / slowest) * 100}%` }} /></span>
            <span className={length.delta_s > 0.15 ? 'slower' : length.delta_s < -0.15 ? 'faster' : ''}>{formatDelta(length.delta_s)}</span>
            <span>{length.cumulative}</span>
            <span>{length.pace_per_100}</span>
            <span>{length.stroke_rate ?? '—'}</span>
          </button>
        ))}
      </section>
      <p className="table-note">These are the wall-to-wall splits from the video. Pace / 100 is how long 100 {report.unit === 'yd' ? 'yards' : 'meters'} would take at that length’s speed. Stroke rate is {report.stroke_rate_unit}. The first length includes your dive. Tap a length to watch it.</p>
      <Notes notes={report.notes} />
    </div>
  )
}

function WorkOnTab({ report }) {
  if (!report.focus?.length) {
    return <div className="empty-panel"><h2>Nothing stood out to fix</h2><p>This race did not show a clear weak point. Keep racing this event and compare the next one.</p></div>
  }
  return (
    <div className="tab-stack">
      <p className="lead-copy flush">Not a practice plan. These are the things from this race you need to work on. Look up sets for each one.</p>
      {report.focus.map((area, index) => (
        <article className="focus-card" key={area.title}>
          <span className="focus-number">{index + 1}</span>
          <div>
            <h2>{area.title}</h2>
            <p className="body-copy">{area.why}</p>
            <p className="focus-how">{area.how}</p>
            {area.look_up?.length > 0 && (
              <div className="look-up">
                <span>You can look up sets for:</span>
                <div>{area.look_up.map((topic) => <a key={topic} href={`https://www.google.com/search?q=${encodeURIComponent(`${topic} swimming workout`)}`} target="_blank" rel="noreferrer">{topic}</a>)}</div>
              </div>
            )}
          </div>
        </article>
      ))}
    </div>
  )
}

function formatInline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => (
    part.startsWith('**') && part.endsWith('**') ? <strong key={index}>{part.slice(2, -2)}</strong> : part
  ))
}

function RichText({ text }) {
  const blocks = []
  let list = null
  text.split('\n').forEach((raw) => {
    const line = raw.trim()
    const item = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/)
    if (item) {
      if (!list) { list = []; blocks.push(list) }
      list.push(item[1])
      return
    }
    list = null
    if (line) blocks.push(line.replace(/^#+\s*/, ''))
  })
  return blocks.map((block, index) => (
    Array.isArray(block)
      ? <ul key={index}>{block.map((entry, itemIndex) => <li key={itemIndex}>{formatInline(entry)}</li>)}</ul>
      : <p key={index}>{formatInline(block)}</p>
  ))
}

function CoachTab({ report }) {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }, [messages, busy])

  const send = async (text) => {
    const cleaned = text.trim()
    if (!cleaned || busy) return
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages((current) => [...current, { role: 'user', content: cleaned }])
    setQuestion('')
    setNotice('')
    setBusy(true)
    try {
      const response = await askCoach(report.analysis_id, cleaned, history)
      setMessages((current) => [...current, { role: 'assistant', content: response.answer }])
    } catch (err) {
      setMessages((current) => current.slice(0, -1))
      setQuestion(cleaned)
      setNotice(err.message)
    } finally { setBusy(false) }
  }

  return (
    <div className="coach-layout">
      <aside>
        <span>Ask about this race</span>
        <h2>Your SwimVision coach</h2>
        <p>A real coach model that can see this race’s splits, start and stroke rate. The first answer can take a little while. If it is offline, add an OpenAI or Gemini key to <code>.env</code>.</p>
        <div className="prompt-list">{report.suggested_questions?.map((prompt) => <button key={prompt} type="button" disabled={busy} onClick={() => send(prompt)}>{prompt}<span>→</span></button>)}</div>
      </aside>
      <section className="chat-panel">
        <div className="messages">
          {messages.length === 0 && <div className="chat-empty"><WaveMark /><p>Ask your first question below.</p></div>}
          {messages.map((message, index) => (
            <div key={`${message.role}-${index}`} className={`message ${message.role === 'assistant' ? 'coach' : 'user'}`}>
              {message.role === 'assistant' && <WaveMark />}
              <div className="bubble">{message.role === 'assistant' ? <RichText text={message.content} /> : <p>{message.content}</p>}</div>
            </div>
          ))}
          {busy && <div className="message coach typing"><WaveMark /><div className="bubble"><p><i /><i /><i /></p></div></div>}
          <div ref={endRef} />
        </div>
        {notice && <p className="form-error" role="alert">{notice}</p>}
        <form onSubmit={(event) => { event.preventDefault(); send(question) }}>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about this race…" maxLength={500} />
          <button type="submit" disabled={busy || !question.trim()}>Send</button>
        </form>
      </section>
    </div>
  )
}

const TABS = ['Overview', 'Technique', 'Race', 'What to work on', 'Ask coach']


function ResultsScreen({ report, onNew }) {
  const [tab, setTab] = useState('Overview')
  const videoRef = useRef(null)
  const timeLabel = { official: 'Official time', video: 'Race time' }[report.race_time_source] || 'Race time'
  return (
    <div className="page-shell results-page">
      <Header onNew={onNew} compact />
      <main className="results-shell">
        <section className="result-hero">
          <div>
            <div className="eyebrow"><span /> Analysis complete</div>
            <h1>{report.race_label}</h1>
          </div>
          <div className="time-card">
            <span>{timeLabel}</span>
            <strong>{report.race_time ?? '—'}</strong>
            {report.race_time_source === 'video' && <small>Timed from the video</small>}
          </div>
        </section>
        <RaceVideo report={report} videoRef={videoRef} />
        <nav className="result-tabs" aria-label="Analysis sections">
          {TABS.map((item) => <button type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)} key={item}>{item}</button>)}
        </nav>
        <div className="tab-content">
          {tab === 'Overview' && <OverviewTab report={report} onTab={setTab} />}
          {tab === 'Technique' && <TechniqueTab report={report} />}
          {tab === 'Race' && <RaceTab report={report} videoRef={videoRef} />}
          {tab === 'What to work on' && <WorkOnTab report={report} />}
          {tab === 'Ask coach' && <CoachTab report={report} />}
        </div>
      </main>
    </div>
  )
}

function App() {
  const [screen, setScreen] = useState('upload')
  const [analysis, setAnalysis] = useState(null)
  const [report, setReport] = useState(null)
  const [fatalError, setFatalError] = useState('')

  const reset = () => { setScreen('upload'); setAnalysis(null); setReport(null); setFatalError('') }
  const complete = useCallback(async () => {
    try { setReport(await getAnalysis(analysis.analysis_id)); setScreen('results') }
    catch (err) { setFatalError(err.message); setScreen('error') }
  }, [analysis])
  const fail = useCallback((message) => { setFatalError(message); setScreen('error') }, [])

  if (screen === 'select') return <SelectScreen analysis={analysis} onBack={reset} onConfirmed={() => setScreen('processing')} />
  if (screen === 'processing') return <ProcessingScreen analysisId={analysis.analysis_id} onComplete={complete} onFail={fail} onNew={reset} />
  if (screen === 'results') return <ResultsScreen report={report} onNew={reset} />
  if (screen === 'error') return <div className="page-shell"><Header onNew={reset} compact /><main className="error-page"><span>We couldn’t finish this analysis</span><h1>Let’s try that again.</h1><p>{fatalError}</p><button className="primary-button compact-button" type="button" onClick={reset}>Start over</button></main></div>
  return <UploadScreen onCreated={(created) => { setAnalysis(created); setScreen('select') }} />
}

export default App
