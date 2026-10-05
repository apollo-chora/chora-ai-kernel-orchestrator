// companion_bus_wiring.go - ADR-254 D4/D8 composition root for the two
// caller-facing workflow lanes consumption owns:
//
//	companion_turn        every Companion turn (typed chat, skill invoke,
//	                      ceremony edge scout, ritual step, dose greeting)
//	dose_recommendation   the daily-dose AI picks
//
// Outbound: the BusTurnEngine / BusDoseRecommender write the turn row and the
// outbox REQUEST row in ONE transaction (pg CompanionTurnRepo.Begin over the
// ambient querier + TxOutbox.PublishGrownInTx); the existing outbox dispatcher
// publishes the request to chora.consumption.companion_turn.requested.v1 /
// dose_recommendation.requested.v1 (JSON-wire, schemaless).
//
// Inbound: two PULL subscriptions (provisioned by the coordinator's
// companion_topic_estate.tf; ids env-overridable) feed the completed-handler,
// which completes the stored turn; the HTTP waiter polls the store, so any
// replica may receive the completion.
//
// Env (read ONLY here, feedback_no_inline_config):
//
//	CHORA_COMPANION_TURN_COMPLETED_SUBSCRIPTION       default chora-consumption.companion-turn-completed
//	CHORA_DOSE_RECOMMENDATION_COMPLETED_SUBSCRIPTION  default chora-consumption.dose-recommendation-completed
//	CHORA_COMPANION_TURN_DEADLINE                     default 130s (kennel typed-chat park deadline 120s + margin)
//	AI_KERNEL_ORCHESTRATOR_URL                        ADR-197 P3 prompt-override resolver base (unset = baseline renders)
//
// Fail-loud: without a pg pool the engines are NOT wired (the chat / skill /
// dose-AI handlers answer 503 ENGINE_NOT_CONFIGURED); without a Pub/Sub client
// the request rows still queue in the outbox but no completion can ever arrive,
// so the wiring logs that loudly and the waiter times every turn out
// (COMPANION_TURN_TIMEOUT), never silence.
package main

import (
	"context"
	"errors"
	"log"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/apollo-chora/chora-common/eventbus"

	"github.com/apollo-chora/chora-consumption/internal/adapter/clients"
	"github.com/apollo-chora/chora-consumption/internal/adapter/events"
	httpadapter "github.com/apollo-chora/chora-consumption/internal/adapter/http"
	consumptionpg "github.com/apollo-chora/chora-consumption/internal/adapter/repo/pg"
	"github.com/apollo-chora/chora-consumption/internal/domain/companion"
)

const (
	envCompanionTurnCompletedSub      = "CHORA_COMPANION_TURN_COMPLETED_SUBSCRIPTION"
	envDoseRecommendationCompletedSub = "CHORA_DOSE_RECOMMENDATION_COMPLETED_SUBSCRIPTION"
	envCompanionTurnDeadline          = "CHORA_COMPANION_TURN_DEADLINE"
	envPromptRegistryBase             = "AI_KERNEL_ORCHESTRATOR_URL"
	outboxAggregateCompanionTurn      = "companion_turn"
)

// wireCompanionBus builds the bus-backed engines onto srv and starts the pull
// consumers. Idempotent on a nil pool (nothing wired, loud log).
func wireCompanionBus(ctx context.Context, srv *httpadapter.Server, pool *pgxpool.Pool, bus eventbus.Bus) {
	if pool == nil {
		log.Printf("consumption: companion bus NOT wired (CHORA_DB_DSN unset: no turn store) - chat / skill invoke / dose AI answer 503 ENGINE_NOT_CONFIGURED")
		return
	}
	store := consumptionpg.NewCompanionTurnRepo(consumptionpg.NewPgxTxRunner(pool))
	txOutbox := consumptionpg.NewTxOutbox(outboxAggregateCompanionTurn)
	publish := clients.TurnRequestPublisherFunc(txOutbox.PublishGrownInTx)

	var promptOverrides clients.PromptOverridesResolver
	if base := strings.TrimSpace(os.Getenv(envPromptRegistryBase)); base != "" {
		// ADR-254 D9: the prompt-registry agent id is the crew id companion_chat
		// (/v1/prompt-registry/agents/companion_chat/resolve), fail-open on a miss.
		promptOverrides = clients.NewPromptRegistryHTTPResolver(base, "companion_chat", 60*time.Second, nil)
		log.Printf("consumption: companion prompt override resolver wired (registry=%s)", base)
	} else {
		log.Printf("consumption: %s unset - companion prompt overrides disabled (embedded baseline renders)", envPromptRegistryBase)
	}

	deadline := clients.DefaultTurnDeadline
	if v := strings.TrimSpace(os.Getenv(envCompanionTurnDeadline)); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			log.Fatalf("consumption: %s=%q invalid (want a positive Go duration) - fail-loud per feedback_no_inline_config", envCompanionTurnDeadline, v)
		}
		deadline = d
	}

	engine, err := clients.NewBusTurnEngine(clients.BusTurnEngineConfig{
		Store:           store,
		Publish:         publish,
		PromptOverrides: promptOverrides,
		TurnDeadline:    deadline,
	})
	if err != nil {
		log.Fatalf("consumption: companion bus turn engine: %v", err)
	}
	srv.CompanionEngine = engine
	srv.CompanionEngineResource = "bus:" + events.TopicCompanionTurnRequested

	dose, err := clients.NewBusDoseRecommender(clients.BusDoseRecommenderConfig{
		Store:   store,
		Publish: publish,
	})
	if err != nil {
		log.Fatalf("consumption: dose recommendation lane: %v", err)
	}
	srv.RecommenderEngine = dose
	srv.RecommenderEngineResource = "bus:" + events.TopicDoseRecommendationRequested
	log.Printf("consumption: companion bus wired (turn deadline %s): chat/skills/ceremony/rituals/greeting -> %s, dose AI -> %s",
		deadline, events.TopicCompanionTurnRequested, events.TopicDoseRecommendationRequested)

	if bus == nil {
		log.Printf("ERROR consumption: companion bus has NO event bus (NATS_URL unset) - requests queue in the outbox but no completion can arrive; every turn will time out (%s)", companion.TurnErrCodeTimeout)
		return
	}
	startCompletedConsumer(ctx, bus, store, companion.TurnLaneCompanionTurn,
		envCompanionTurnCompletedSub, events.SubscriptionCompanionTurnCompleted, events.TopicCompanionTurnCompleted)
	startCompletedConsumer(ctx, bus, store, companion.TurnLaneDoseRecommendation,
		envDoseRecommendationCompletedSub, events.SubscriptionDoseRecommendationCompleted, events.TopicDoseRecommendationCompleted)
}

func startCompletedConsumer(ctx context.Context, bus eventbus.Bus, store companion.TurnStore, lane companion.TurnLane, envKey, defaultSub, topic string) {
	sub := strings.TrimSpace(os.Getenv(envKey))
	if sub == "" {
		sub = defaultSub
	}
	handler := events.NewCompanionTurnCompletedHandler(store, lane)
	go func() {
		log.Printf("consumption: %s result consumer binding %s -> %s", lane, sub, topic)
		if err := bus.Subscribe(ctx, consumerConfig(sub, topic), handler); err != nil && !errors.Is(err, context.Canceled) {
			// A consumer that exits is a lane that silently stops completing
			// turns: say so at ERROR level (the waiter surfaces TIMEOUT).
			log.Printf("ERROR consumption: %s result consumer exited: %v", lane, err)
		}
	}()
}
