package boot

import (
	"fmt"
	"iter"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/5007-Capstone/chora/agents/chora-adk-common/agentdispatch"
	"github.com/5007-Capstone/chora/agents/chora-adk-common/promptstamping"

	dagent "github.com/5007-Capstone/chora/agents/companion-diagnosis-adk-go/internal/agent"
)

// Session-state keys the dispatch payload carries (agentdispatch merges
// input_payload into session state wholesale, ADR-254 D6 addendum contract).
const (
	stateKeyTaskKind      = "task_kind"
	stateKeyExtractedText = "extracted_text"
	stateKeyCluesBlock    = "clues_block"
	stateKeyEdgesJSON     = "edges_json"
	stateKeyMaxQuestions  = "max_questions"
	stateKeySourceBlobURI = "source_blob_uri"
	stateKeySourceMime    = "source_mime_type"
	stateKeyArtifactB64   = "artifact_b64"
	// Practice-test loop state (written by OutputKey and the gate's StateDelta).
	stateKeyActorRaw    = "practice_actor_raw"
	stateKeyCriticRaw   = "practice_critic_raw"
	stateKeyPriorNotes  = "practice_prior_notes"
	stateKeyRound       = "practice_round"
	stateKeyCandidatesN = "practice_candidates_n"
)

// Agent tree names. The terminal text for both roles is the LAST text event
// from ANY author (agentdispatch.TerminalAuthor is empty for them), so the
// only rule the tree must keep is: the final answer is the last text event.
const (
	singleTaskAgentName = "companion_diagnoser_task"
	practiceAgentName   = "companion_diagnoser_practice"
	actorAgentName      = "companion_diagnoser_practice_actor"
	criticAgentName     = "companion_diagnoser_practice_critic"
	gateAgentName       = "companion_diagnoser_practice_gate"
)

type stateReader interface {
	Get(string) (any, error)
}

// ---------------------------------------------------------------------------
// extractor
// ---------------------------------------------------------------------------

// extractorInstructionProvider composes the transcription prompt from the
// artifact reference. The bus carries a gs:// reference only: a missing one is
// permanent (no redelivery can supply it), and an inline artifact_b64 is
// refused by name rather than silently ignored, because the grounding plugin
// would never put it in front of the model.
func extractorInstructionProvider() llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		st := ctx.ReadonlyState()
		blobURI := strings.TrimSpace(StateString(st, stateKeySourceBlobURI))
		if blobURI == "" {
			if strings.TrimSpace(StateString(st, stateKeyArtifactB64)) != "" {
				return "", agentdispatch.Permanent("unsupported_artifact_source: artifact_b64 is not carried on the bus; dispatch source_blob_uri (gs://)", nil)
			}
			return "", agentdispatch.Permanent("missing_source_blob_uri: the extract dispatch carries no gs:// artifact reference", nil)
		}
		if !strings.HasPrefix(blobURI, "gs://") {
			return "", agentdispatch.Permanent("invalid_source_blob_uri: not a gs:// reference", fmt.Errorf("source_blob_uri=%q", blobURI))
		}
		return dagent.ComposeExtractorInstruction(StateString(st, stateKeySourceMime), blobURI), nil
	}
}

// NewExtractorAgent builds the companion_extractor root llmagent.
func NewExtractorAgent(cfg Config, llm adkmodel.LLM) (agent.Agent, error) {
	a, err := llmagent.New(llmagent.Config{
		Name:  CrewKindExtractor,
		Model: llm,
		InstructionProvider: promptstamping.WithStamping(
			cfg.PromptVersion,
			func(s session.ReadonlyState) map[string]string { return dagent.ExtractorConditions(s) },
			extractorInstructionProvider(),
		),
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", CrewKindExtractor, err)
	}
	return a, nil
}

// ---------------------------------------------------------------------------
// diagnoser: task_kind router over a single-shot llmagent and a practice loop
// ---------------------------------------------------------------------------

// selectTask reads and validates task_kind from session state.
func selectTask(st stateReader) (dagent.TaskKind, error) {
	kind, err := dagent.ParseTaskKind(StateString(st, stateKeyTaskKind))
	if err != nil {
		return "", agentdispatch.Permanent(err.Error(), nil)
	}
	return kind, nil
}

// singleTaskInstructionProvider composes the diagnose or study_aids prompt.
// practice_test never reaches it (the router sends that to the loop).
func singleTaskInstructionProvider() llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		st := ctx.ReadonlyState()
		kind, err := selectTask(st)
		if err != nil {
			return "", err
		}
		switch kind {
		case dagent.TaskDiagnose:
			extracted := StateString(st, stateKeyExtractedText)
			if strings.TrimSpace(extracted) == "" {
				return "", agentdispatch.Permanent("missing_extracted_text: the diagnose dispatch carries no extractor output", nil)
			}
			return dagent.ComposeDiagnoserInstruction(extracted, StateString(st, stateKeyCluesBlock)), nil
		case dagent.TaskStudyAids:
			edges, err := dagent.ParseEdges(StateString(st, stateKeyEdgesJSON))
			if err != nil {
				return "", agentdispatch.Permanent("invalid_edges_json", err)
			}
			return dagent.ComposeStudyAidsInstruction(edges), nil
		}
		return "", agentdispatch.Permanent("unsupported_task_kind: "+string(kind)+" is not a single-shot task", nil)
	}
}

// actorInstructionProvider composes the practice-test actor prompt from the
// edges, the cap and the previous round's critic notes.
func actorInstructionProvider() llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		st := ctx.ReadonlyState()
		edges, err := dagent.ParseEdges(StateString(st, stateKeyEdgesJSON))
		if err != nil {
			return "", agentdispatch.Permanent("invalid_edges_json", err)
		}
		maxQ := dagent.MaxQuestionsFrom(StateString(st, stateKeyMaxQuestions))
		return dagent.ComposeActorInstruction(edges, maxQ, StateString(st, stateKeyPriorNotes)), nil
	}
}

// practiceCandidates reads the actor's last answer and normalises it on-edge.
func practiceCandidates(st stateReader) ([]dagent.Question, []dagent.Edge, error) {
	edges, err := dagent.ParseEdges(StateString(st, stateKeyEdgesJSON))
	if err != nil {
		return nil, nil, err
	}
	maxQ := dagent.MaxQuestionsFrom(StateString(st, stateKeyMaxQuestions))
	return dagent.ParseCandidates(StateString(st, stateKeyActorRaw), dagent.KnownKeys(edges), maxQ), edges, nil
}

// criticInstructionProvider composes the critic prompt over the candidates.
func criticInstructionProvider() llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		candidates, _, err := practiceCandidates(ctx.ReadonlyState())
		if err != nil {
			return "", agentdispatch.Permanent("invalid_edges_json", err)
		}
		return dagent.ComposeCriticInstruction(candidates), nil
	}
}

// skipCriticWithoutCandidates is the critic's BeforeAgentCallback: when the
// actor produced no on-edge candidate there is nothing to critique, so the
// model call is skipped and an empty verdict list stands in (the gate then
// ends the run with the actor's failure, never a regenerate).
func skipCriticWithoutCandidates(ctx agent.CallbackContext) (*genai.Content, error) {
	candidates, _, err := practiceCandidates(ctx.State())
	if err != nil {
		return nil, agentdispatch.Permanent("invalid_edges_json", err)
	}
	if len(candidates) > 0 {
		return nil, nil
	}
	return &genai.Content{Role: "model", Parts: []*genai.Part{{Text: `{"verdicts":[]}`}}}, nil
}

// gateOutcome is the gate's decision for one loop round, rendered as an event.
type gateOutcome struct {
	Text       string         // the final answer when Escalate; "" on a regenerate
	Escalate   bool           // ends the loop
	StateDelta map[string]any // regenerate guidance for the next actor turn
}

// decidePractice is the pure gate: accepted questions end the loop with the
// rendered test; an all-rejected round regenerates once with the critic's
// notes; a non-producing actor or an exhausted budget ends with an empty,
// reasoned result (OK on the wire, the output is opt-in and best-effort).
func decidePractice(st stateReader) (gateOutcome, error) {
	candidates, edges, err := practiceCandidates(st)
	if err != nil {
		return gateOutcome{}, agentdispatch.Permanent("invalid_edges_json", err)
	}
	round := 0
	if r, ok := st.Get(stateKeyRound); ok == nil {
		switch v := r.(type) {
		case int:
			round = v
		case float64:
			round = int(v)
		}
	}
	if len(candidates) == 0 {
		return gateOutcome{Text: dagent.RenderPracticeTest(edges, nil,
			"actor produced no on-edge question"), Escalate: true}, nil
	}
	verdicts := dagent.ParseVerdicts(StateString(st, stateKeyCriticRaw))
	var accepted []dagent.Question
	notes := "critic verdict unreadable"
	if verdicts != nil {
		accepted, notes = dagent.Gate(candidates, verdicts)
	}
	if len(accepted) > 0 {
		return gateOutcome{Text: dagent.RenderPracticeTest(edges, accepted, ""), Escalate: true}, nil
	}
	if round < dagent.MaxRegenRounds {
		if notes == "" {
			notes = "every candidate was rejected"
		}
		return gateOutcome{StateDelta: map[string]any{
			stateKeyPriorNotes: notes, stateKeyRound: round + 1, stateKeyCandidatesN: len(candidates),
		}}, nil
	}
	reason := fmt.Sprintf("critic rejected every candidate in %d rounds", round+1)
	if notes != "" {
		reason += ": " + notes
	}
	return gateOutcome{Text: dagent.RenderPracticeTest(edges, nil, reason), Escalate: true}, nil
}

// newGateAgent wraps decidePractice as the loop's last sub-agent.
func newGateAgent() (agent.Agent, error) {
	return agent.New(agent.Config{
		Name:        gateAgentName,
		Description: "deterministic on-edge gate over the critic's verdicts",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				out, err := decidePractice(ctx.Session().State())
				if err != nil {
					yield(nil, err)
					return
				}
				ev := session.NewEvent(ctx.InvocationID())
				ev.Author = gateAgentName
				if out.Text != "" {
					ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{Text: out.Text}}}
				}
				ev.Actions.Escalate = out.Escalate
				if len(out.StateDelta) > 0 {
					ev.Actions.StateDelta = out.StateDelta
				}
				yield(ev, nil)
			}
		},
	})
}

// newPracticeAgent builds the bounded actor -> critic -> gate loop.
func newPracticeAgent(llm adkmodel.LLM) (agent.Agent, error) {
	actor, err := llmagent.New(llmagent.Config{
		Name: actorAgentName, Model: llm, OutputKey: stateKeyActorRaw,
		InstructionProvider: actorInstructionProvider(),
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", actorAgentName, err)
	}
	critic, err := llmagent.New(llmagent.Config{
		Name: criticAgentName, Model: llm, OutputKey: stateKeyCriticRaw,
		InstructionProvider:  criticInstructionProvider(),
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{skipCriticWithoutCandidates},
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", criticAgentName, err)
	}
	gate, err := newGateAgent()
	if err != nil {
		return nil, fmt.Errorf("agent.New(%s): %w", gateAgentName, err)
	}
	loop, err := loopagent.New(loopagent.Config{
		AgentConfig: agent.Config{
			Name:        practiceAgentName,
			Description: "practice test: actor, critic, on-edge gate, one bounded regenerate",
			SubAgents:   []agent.Agent{actor, critic, gate},
		},
		MaxIterations: uint(1 + dagent.MaxRegenRounds),
	})
	if err != nil {
		return nil, fmt.Errorf("loopagent.New(%s): %w", practiceAgentName, err)
	}
	return loop, nil
}

// NewDiagnoserAgent builds the companion_diagnoser root: a task_kind router
// over the single-shot task agent (diagnose, study_aids) and the practice
// loop. An unknown task_kind is a permanent failure raised before any model
// call, reported on the wire as FAILED unknown_task_kind.
func NewDiagnoserAgent(cfg Config, llm adkmodel.LLM) (agent.Agent, error) {
	single, err := llmagent.New(llmagent.Config{
		Name:  singleTaskAgentName,
		Model: llm,
		InstructionProvider: promptstamping.WithStamping(
			cfg.PromptVersion,
			func(s session.ReadonlyState) map[string]string { return diagnoserConditions(s) },
			singleTaskInstructionProvider(),
		),
	})
	if err != nil {
		return nil, fmt.Errorf("llmagent.New(%s): %w", singleTaskAgentName, err)
	}
	practice, err := newPracticeAgent(llm)
	if err != nil {
		return nil, err
	}
	return agent.New(agent.Config{
		Name:        CrewKindDiagnoser,
		Description: "companion_diagnose task_kind router",
		SubAgents:   []agent.Agent{single, practice},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				kind, err := selectTask(ctx.Session().State())
				if err != nil {
					yield(nil, err)
					return
				}
				child := single
				if kind == dagent.TaskPracticeTest {
					child = practice
				}
				for ev, err := range child.Run(ctx) {
					if !yield(ev, err) {
						return
					}
				}
			}
		},
	})
}

// diagnoserConditions surfaces the prompt discriminants (ADR-197 M-A): the
// task kind and, for diagnose, whether clues were threaded in.
func diagnoserConditions(st stateReader) map[string]string {
	cond := dagent.DiagnoserConditions(st)
	kind, err := dagent.ParseTaskKind(StateString(st, stateKeyTaskKind))
	if err != nil {
		kind = "unknown"
	}
	cond["task_kind"] = string(kind)
	return cond
}
