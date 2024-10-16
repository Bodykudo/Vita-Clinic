import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from 'src/prisma.service';

import {
  CreateScanDto,
  GetPatientScansQuery,
  UpdateScanDto,
} from './dto/scans.dto';
import { LogService } from 'src/log/log.service';

@Injectable()
export class ScansService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly logService: LogService,
  ) {}

  async findAllByPatientId(
    patientId: string,
    {
      page = 1,
      limit = 10,
      value = '',
      sort = 'createdAt-desc',
    }: GetPatientScansQuery,
  ) {
    const [sortField, sortOrder] = sort.split('-') as [string, 'desc' | 'asc'];

    return this.prisma.scan.findMany({
      where: {
        appointment: { patientId },
        title: { contains: value, mode: 'insensitive' },
      },
      select: {
        id: true,
        title: true,
        notes: true,
        createdAt: true,
        updatedAt: true,
        modality: true,
        appointment: true,
      },
      skip: (page - 1) * limit,
      take: limit,
      orderBy: [
        {
          title: sortField === 'name' ? sortOrder : undefined,
        },
        {
          createdAt: sortField === 'createdAt' ? sortOrder : undefined,
        },
      ],
    });
  }

  async findAllByAppointmentId(appointmentId: string) {
    return this.prisma.scan.findMany({
      where: { appointmentId },
      select: {
        id: true,
        title: true,
        notes: true,
        createdAt: true,
        updatedAt: true,
        modality: true,
        appointment: true,
      },
    });
  }

  async findById(id: string) {
    const scan = await this.prisma.scan.findUnique({
      where: { id },
      include: {
        appointment: true,
        modality: true,
      },
    });

    if (!scan) {
      throw new NotFoundException('Scan not found');
    }

    return scan;
  }

  async create(createScanDto: CreateScanDto, userId: string) {
    const { title, notes, modalityId, scanURLs, appointmentId } = createScanDto;

    const appointment = await this.prisma.appointment.findUnique({
      where: { id: appointmentId },
    });

    const modality = await this.prisma.modality.findUnique({
      where: { id: modalityId },
    });

    if (!appointment || !modality) {
      throw new NotFoundException('Appointment not found');
    }

    const scan = await this.prisma.scan.create({
      data: {
        title,
        notes,
        modality: { connect: { id: modalityId } },
        scanURLs,
        appointment: { connect: { id: appointmentId } },
      },
    });

    await this.logService.create({
      userId,
      targetId: scan.id,
      targetName: scan.title,
      type: 'scan',
      action: 'create',
      targetUserId: appointment.patientId,
    });

    return scan;
  }

  async update(id: string, updateScanDto: UpdateScanDto, userId: string) {
    const { title, notes } = updateScanDto;

    const existingScan = await this.prisma.scan.findUnique({
      where: { id },
    });

    if (!existingScan) {
      throw new NotFoundException('Scan not found');
    }

    const scan = await this.prisma.scan.update({
      where: { id },
      data: {
        title,
        notes,
      },
      include: {
        appointment: {
          select: {
            patientId: true,
          },
        },
      },
    });

    await this.logService.create({
      userId,
      targetId: scan.id,
      targetName: scan.title,
      type: 'scan',
      action: 'update',
      targetUserId: scan.appointment.patientId,
    });
  }
}
