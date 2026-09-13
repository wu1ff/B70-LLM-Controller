package runtime

import (
	"context"
	"os/exec"
)

var runDocker = func(args ...string) ([]byte, error) {
	return exec.Command("docker", args...).CombinedOutput()
}

var runDockerPull = streamDockerPull

func streamDockerPull(ctx context.Context, reference string, notify func(PullProgress)) ([]byte, error) {
	command := exec.CommandContext(ctx, "docker", "pull", reference)
	writer := newDockerProgressWriter(notify)
	command.Stdout = writer
	command.Stderr = writer
	err := command.Run()
	writer.flush()
	return []byte(writer.summary()), err
}
