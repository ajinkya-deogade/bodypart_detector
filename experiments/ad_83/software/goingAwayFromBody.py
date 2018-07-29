## Compute the allocentricHeadAngle
midHeadVector = headPosition - midPosition
AllocentricHeadAngle = atan2(midHeadVector(:, 2), midHeadVector(:, 1)).*(180 / pi)
history = 1000

## Smoothen the allocentricHeadAngle
smoothedAllocentricHeadAngle = smoothen(AllocentricHeadAngle, history)

## Compute the allocentricHeadAngleSpeed
smoothedAllocentricHeadAngleSpeed = unwrap(smoothedAllocentricHeadAngle)
smoothedAllocentricHeadAngleSpeed = diff(smoothedAllocentricHeadAngle) ## Difference from the previous frame

## Compute the headsweeps by applying a threshold on the smoothedAllocentricHeadAngleSpeed
temp = zeros(length(smoothedAllocentricHeadAngleSpeed), 1)
temp(isnan(smoothedAllocentricHeadAngleSpeed)) = 0
temp(abs(smoothedAllocentricHeadAngleSpeed) > thresh) = 1
diffTemp = diff(temp)
changeIndexPositive = find(diffTemp > 0)
changeIndexNegative = find(diffTemp < 0)

## Call the ones above threshold as putative sweeps
headSweeps = [changeIndexPositive(1:length(changeIndexNegative)), changeIndexNegative]


if not empty(headSweeps)
    for swp = 1:size(headSweeps, 1)
        ## Look at the head angles with in a head sweep for zero crossings
        headAngleSweep = []
        headAngleSweep = headAngleAll(headSweeps(swp, 1):headSweeps(swp, 2))
        signChanges = find(diff(headAngleSweep > 0))
        if not empty(signChanges)
            ## Break down the sweeps for going away and going the other side
            for i = 2:length(headAngleSweep)
                if ((headAngleSweep(i - 1) < 0) & (headAngleSweep(i) > 0)) | ((headAngleSweep(i - 1) > 0) & (headAngleSweep(i) < 0))
                    sweepsBroken(end + 1, 1) = headSweeps(swp, 1) + (find(headAngleSweep == headAngleSweep(1)) - 1)
                    sweepsBroken(end, 2) = headSweeps(swp, 1) + (find(headAngleSweep == headAngleSweep(i - 1)) - 1)
                    sweepsBroken(end + 1, 1) = headSweeps(swp, 1) + (find(headAngleSweep == headAngleSweep(i)) - 1)
                    sweepsBroken(end, 2) = headSweeps(swp, 1) + (find(headAngleSweep == headAngleSweep(end)) - 1)
                else
                    sweepsBroken(end + 1,:) = headSweeps(swp, 1:2)


## Clean up the sweeps to avoid jitter by having a threshold on the mean head angle within a sweep 'thresholdToMakeClean'
headAngleChange = diff(abs(headAngleAll))
sweepsFinal = []
for swp = 1:size(sweepsBroken, 1)
    if sweepsBroken(swp, 2) <= length(headAngleChange)
        sweepsBroken(swp, 3) = mean(headAngleChange(sweepsBroken(swp, 1):sweepsBroken(swp, 2)))
        if sweepsBroken(swp, 3) > thresholdToMakeClean % ensure going outward only
            sweepsFinal(end + 1,:) = sweepsBroken(swp,:)


minimumCountThresholdCrossings = get from user ## minimum number of ThresholdCrossings
LeftIntensityVector = get from user
RightIntensityVector = get from user
LeftHeadAngleThreshold = get from user
RightHeadAngleThreshold = get from user

for each frame
    presentHeadAngle = getHeadAngle(frame)
    if entered strictlyRunning
        countGoodSweepWithInSampling = 0
        SelectedLeftIntensity = select randomly from LeftIntensityVector ## (without replacement)
        SelectedRightIntensity = select randomly from RightIntensityVector ## (without replacement)
        presentIntensity = null

        if presentIntensity is not null
            if presentHeadAngle < LeftHeadAngleThreshold and presentHeadAngle > RightHeadAngleThreshold
                apply presentIntensity

        if the presentHeadAngle >= LeftHeadAngleThreshold
            presentIntensity = baseline + SelectedLeftIntensity
            apply presentIntensity
            countThresholdCrossings++
        if the presentHeadAngle <= RightHeadAngleThreshold
            presentIntensity = baseline + SelectedRightIntensity
            apply presentIntensity
            countThresholdCrossings++

    if already strictlyRunning
        if countThresholdCrossings <= minimumCountThresholdCrossings
            if presentIntensity is not null
                if presentHeadAngle < LeftHeadAngleThreshold and presentHeadAngle > RightHeadAngleThreshold
                    apply presentIntensity

            if the presentHeadAngle >= LeftHeadAngleThreshold
                presentIntensity = baseline + SelectedLeftIntensity
                apply presentIntensity
                countThresholdCrossings++
            if the presentHeadAngle <= RightHeadAngleThreshold
                presentIntensity = baseline + SelectedRightIntensity
                apply presentIntensity
                countThresholdCrossings++
